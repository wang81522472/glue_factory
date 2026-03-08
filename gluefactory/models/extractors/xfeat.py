"""XFeat feature extractor integration for glue-factory.
This module integrates the XFeat feature extractor from accelerated_features into the glue-factory framework.

Authors: XFeat - https://github.com/verlab/accelerated_features
Integration: glue-factory team
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf
import importlib.util

# 处理导入方式，区分直接运行和作为模块导入
if __name__ == "__main__":
    # 当直接运行此文件时使用绝对导入
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from gluefactory.models.base_model import BaseModel
    from gluefactory.models.utils.misc import pad_to_length, pad_and_stack
    from gluefactory.geometry.homography import warp_points_torch, distort_points_fisheye
    from gluefactory.settings import DATA_PATH
else:
    # 当作为模块导入时使用相对导入
    from ..base_model import BaseModel
    from ..utils.misc import pad_to_length, pad_and_stack
    from ...geometry.homography import warp_points_torch, distort_points_fisheye
    from ...settings import DATA_PATH

# 使用相对路径查找accelerated_features
def find_accelerated_features():
    """动态查找accelerated_features库路径"""
    # 首先检查环境变量
    if 'ACCELERATED_FEATURES_PATH' in os.environ:
        return os.environ['ACCELERATED_FEATURES_PATH']
    
    # 然后尝试在当前目录的平行目录中查找
    current_file = Path(__file__).resolve()
    # 获取glue-factory根目录
    glue_factory_root = current_file.parent.parent.parent.parent
    # 检查平行目录
    accelerated_features_dir = glue_factory_root.parent / "accelerated_features"
    if accelerated_features_dir.exists():
        print(f"Found accelerated_features at {accelerated_features_dir}")
        return str(accelerated_features_dir)
    
    # 如果上述方法都失败，尝试在Python路径中查找
    for path in sys.path:
        potential_path = Path(path) / "accelerated_features"
        if potential_path.exists():
            return str(potential_path)
    
    # 最后回退到原来的硬编码路径
    fallback_path = "/home/dji/workspace/project_UNO/accelerated_features"
    if Path(fallback_path).exists():
        return fallback_path
    
    raise ImportError(
        "Cannot find accelerated_features. Please set ACCELERATED_FEATURES_PATH "
        "environment variable or ensure it's installed in a standard location."
    )

# Lazy import of accelerated_features - only when XFeat class is actually used
OriginalXFeat = None
InterpolateSparse2d = None
accelerated_features_path = None

def _ensure_accelerated_features():
    """Ensure accelerated_features is available and imported."""
    global OriginalXFeat, InterpolateSparse2d, accelerated_features_path
    
    if OriginalXFeat is not None:
        return  # Already imported
    
    try:
        # 查找并添加accelerated_features路径
        accelerated_features_path = find_accelerated_features()
        if accelerated_features_path not in sys.path:
            sys.path.append(accelerated_features_path)
        
        # Import XFeat from accelerated_features (support both package and repo layouts)
        try:
            from accelerated_features.modules.xfeat import XFeat as OriginalXFeat  # type: ignore
            from accelerated_features.modules.interpolator import InterpolateSparse2d  # type: ignore
        except Exception:
            from modules.xfeat import XFeat as OriginalXFeat  # type: ignore
            from modules.interpolator import InterpolateSparse2d  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Cannot import XFeat from accelerated_features. "
            "Make sure the accelerated_features repository is available. "
            f"Error: {e}"
        )


class XFeat(BaseModel):
    default_conf = {
        "weights": None,  # Path to weights file, None for default
        "max_num_keypoints": 4096,  # Maximum number of keypoints (for glue-factory consistency)
        "top_k": None,  # Maximum number of keypoints (will use max_num_keypoints if None)
        "detection_threshold": 0.05,  # Detection threshold
        "force_num_keypoints": False,  # Force exact number of keypoints
        "trainable": False,  # Enable training
        "apply_xfeat_loss": True,  # Apply XFeat-specific losses
        "xfeat_loss_weight": 1.0,  # Weight for XFeat losses
        # Optional: path to an H5 file containing precomputed ALIKED keypoints per image name.
        # If provided, XFeat training will load keypoints on-the-fly when the dataset does not
        # provide them in `view{0,1}.cache.aliked_keypoints`.
        "aliked_keypoint_dir": None,
        "aliked_keypoint_keys": ("aliked_keypoints", "keypoints", "kpts", "points"),
        "aliked_cache_size": 2048,  # number of images to keep in memory (keypoints only)
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        # Ensure accelerated_features is available
        _ensure_accelerated_features()

        # Flag indicating whether we actually train XFeat
        is_trainable = bool(conf.trainable)

        # Path to optional ALIKED keypoints H5 file (used only for distillation)
        aliked_h5_path = None

        if is_trainable:
            aliked_path = conf.get("aliked_keypoint_dir", None)
            if aliked_path is not None:
                p = Path(str(aliked_path))
                if not p.is_absolute():
                    # Prefer DATA_PATH-relative paths (consistent with other gluefactory exports).
                    cand = Path(DATA_PATH) / p
                    p = cand if cand.exists() else p

                if not p.exists():
                    print(f"WARNING: aliked_keypoint_dir '{aliked_path}' does not exist.")
                    print("WARNING: Disabling XFeat training.")
                    is_trainable = False
                else:
                    aliked_h5_path = str(p)

        # Define local loss functions and utilities.
        # These are adapted from the original accelerated_features implementation
        # but live entirely in this file to avoid importing training modules.
        if is_trainable:
            def dual_softmax_loss(X, Y, temp=0.2):
                if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
                    raise RuntimeError(
                        "Error: X and Y shapes must match and be 2D matrices"
                    )
                dist_mat = (X @ Y.t()) * temp
                conf_matrix12 = F.log_softmax(dist_mat, dim=1)
                conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)
                with torch.no_grad():
                    conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
                    conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
                    conf = conf12 * conf21
                target = torch.arange(len(X), device=X.device)
                loss = F.nll_loss(conf_matrix12, target) + F.nll_loss(
                    conf_matrix21, target
                )
                return loss, conf

            def coordinate_classification_loss(coords1, pts1, pts2, conf):
                # pts1/pts2 are in coarse (1/8) coordinates
                with torch.no_grad():
                    coords1_detached = pts1 * 8
                    offsets1_detached = (coords1_detached / 8) - (
                        coords1_detached / 8
                    ).long()
                    offsets1_detached = (offsets1_detached * 8).long()
                    labels1 = offsets1_detached[:, 0] + 8 * offsets1_detached[:, 1]
                coords1_log = F.log_softmax(coords1, dim=-1)
                predicted = coords1.max(dim=-1)[1]
                acc = (labels1 == predicted)
                acc = acc[conf > 0.1]
                acc = (
                    acc.sum() / len(acc)
                    if len(acc) > 0
                    else torch.tensor(0.0, device=coords1.device)
                )
                loss = F.nll_loss(coords1_log, labels1, reduction="none")
                conf = conf / conf.sum()
                loss = (loss * conf).sum()
                return loss * 2.0, acc

            def keypoint_loss(heatmap, target):
                L1_loss = F.l1_loss(heatmap, target)
                return L1_loss * 3.0

            def check_accuracy(X, Y, pts1=None, pts2=None, plot=False):
                # Simple nearest-neighbour accuracy in descriptor space.
                del pts1, pts2, plot  # unused but kept for API compatibility
                with torch.no_grad():
                    dist_mat = X @ Y.t()
                    nn = torch.argmax(dist_mat, dim=1)
                    correct = nn == torch.arange(len(X), device=X.device)
                    acc = correct.float().mean()
                    return acc

            self.dual_softmax_loss = dual_softmax_loss
            self.coordinate_classification_loss = coordinate_classification_loss
            self.keypoint_loss = keypoint_loss
            self.check_accuracy = check_accuracy
            # ALike distillation via external modules is disabled;
            # we only use pre-loaded ALIKED keypoints.
            self.alike_distill_loss = None
            self.use_alike_loss = False
        else:
            # Skip all training-related utilities when trainable is False
            print("Skipping training utilities setup: trainable is False")
            self.dual_softmax_loss = None
            self.coordinate_classification_loss = None
            self.keypoint_loss = None
            self.alike_distill_loss = None
            self.use_alike_loss = False
            self.check_accuracy = None

        
        # Initialize XFeat model
        default_weights_path = str(Path(accelerated_features_path) / 'weights' / 'xfeat.pt')
        weights_path = default_weights_path
        if conf.weights is not None and Path(conf.weights).exists():
            weights_path = conf.weights
        
        # Use max_num_keypoints if top_k is not provided
        top_k = conf.top_k if conf.top_k is not None else conf.max_num_keypoints
        
        # Create XFeat instance
        self.xfeat = OriginalXFeat(
            weights=weights_path,
            top_k=top_k,
            detection_threshold=conf.detection_threshold
        )
        
        # Expose the underlying XFeatModel for training
        self.net = self.xfeat.net
        
        # Set trainable parameters
        if not is_trainable:
            for p in self.net.parameters():
                p.requires_grad = False
        
        # ALIKED distillation uses pre-computed features from dataset cache
        # No need to initialize ALIKED model here - features are loaded from disk
        self.use_aliked_distill = is_trainable
        # This is used only during training and only if cached keypoints are not provided by the dataset.
        self._aliked_h5 = None
        self._aliked_kpts_cache = OrderedDict()
        self._aliked_kpts_cache_size = int(conf.get("aliked_cache_size", 2048))
        self._aliked_keypoint_keys = tuple(conf.get("aliked_keypoint_keys", ("aliked_keypoints", "keypoints", "kpts", "points")))
        self._aliked_h5_path = aliked_h5_path
        
        # Move to GPU by default for better performance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat.to(self.device)

    def _get_aliked_h5(self):
        """Lazily open the ALIKED keypoints H5 file (read-only)."""
        if self._aliked_h5 is not None:
            return self._aliked_h5
        if self._aliked_h5_path is None:
            return None
        try:
            import h5py  # local import to avoid hard dependency at inference time
        except Exception as e:
            raise ImportError(f"h5py is required to read ALIKED keypoints: {e}")
        if not Path(self._aliked_h5_path).exists():
            # Keep it disabled if the file is missing.
            return None
        self._aliked_h5 = h5py.File(self._aliked_h5_path, "r")
        return self._aliked_h5

    def _load_aliked_keypoints_for_name(self, name: str):
        """Load ALIKED keypoints (N,2) for a given sample name from H5."""
        if name in self._aliked_kpts_cache:
            k = self._aliked_kpts_cache.pop(name)
            self._aliked_kpts_cache[name] = k
            return k

        h5 = self._get_aliked_h5()
        if h5 is None:
            return None

        if name not in h5:
            return None
        grp = h5[name]

        ds = None
        if hasattr(grp, "keys"):
            for k in self._aliked_keypoint_keys:
                if k in grp:
                    ds = grp[k]
                    break
        if ds is None:
            return None

        arr = np.asarray(ds[...], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] != 2:
            return None

        # LRU cache
        self._aliked_kpts_cache[name] = arr
        if len(self._aliked_kpts_cache) > self._aliked_kpts_cache_size:
            self._aliked_kpts_cache.popitem(last=False)
        return arr

    def _transform_aliked_keypoints_to_view(self, kpts_xy: torch.Tensor, view: dict, b: int, inverse: bool = False) -> torch.Tensor:
        """
        Transform ALIKED keypoints from original image coordinates to a view's image coordinates.
        Applies homography (and optional fisheye distortion) and filters to valid image bounds.
        """
        if kpts_xy.numel() == 0:
            return kpts_xy

        # Homography warp: original -> view
        if "H_" in view:
            H = view["H_"][b : b + 1]  # (1,3,3)
            kpts_xy = warp_points_torch(kpts_xy[None], H, inverse=inverse)[0]

        # Optional fisheye distortion (if present in the view dict)
        if "fisheye_params" in view and view["fisheye_params"] is not None:
            try:
                K = view["fisheye_params"]["K"][b] if isinstance(view["fisheye_params"]["K"], torch.Tensor) and view["fisheye_params"]["K"].dim() == 3 else view["fisheye_params"]["K"]
                D = view["fisheye_params"]["D"][b] if isinstance(view["fisheye_params"]["D"], torch.Tensor) and view["fisheye_params"]["D"].dim() == 2 else view["fisheye_params"]["D"]
                pts = distort_points_fisheye(kpts_xy[:, None, :], K, D)  # (N,1,2)
                kpts_xy = pts[:, 0, :]
            except Exception:
                # If fisheye params are malformed, skip distortion.
                pass

        # Filter to image bounds
        img = view.get("image", None)
        if isinstance(img, torch.Tensor):
            H_img, W_img = int(img.shape[-2]), int(img.shape[-1])
            valid = (
                (kpts_xy[:, 0] >= 0.0)
                & (kpts_xy[:, 0] <= float(W_img - 1))
                & (kpts_xy[:, 1] >= 0.0)
                & (kpts_xy[:, 1] <= float(H_img - 1))
            )
            kpts_xy = kpts_xy[valid]
        return kpts_xy

    def _forward(self, data):
        image = data["image"]
        device = image.device
        
        # Move model to the correct device if needed
        if device != self.device:
            self.device = device
            self.xfeat.to(device)
            self.xfeat.dev = device
        
        # 使用批处理模式提取特征
        top_k = self.conf.top_k if self.conf.top_k is not None else self.conf.max_num_keypoints
        outputs = self.xfeat.detectAndCompute(
            image, 
            top_k=top_k,
            detection_threshold=self.conf.detection_threshold
        )
        
        # 收集每个图像的特征点、分数和描述符
        keypoints = []
        scores = []
        descriptors = []
        
        for output in outputs:
            keypoints.append(output['keypoints'])
            scores.append(output['scores'])
            descriptors.append(output['descriptors'])  # Match glue-factory format
        
        # 处理force_num_keypoints选项
        if self.conf.force_num_keypoints:
            num_points = top_k
            keypoints = pad_and_stack(
                keypoints,
                num_points,
                -2,
                mode="random_c",
                bounds=(
                    0,
                    data.get("image_size", torch.tensor(image.shape[-2:])).min().item(),
                ),
            )
            scores = pad_and_stack(
                scores, 
                num_points, 
                -1, 
                mode="zeros"
            )
            descriptors = pad_and_stack(
                descriptors, 
                num_points, 
                -2, 
                mode="zeros"
            )
        else:
            # 如果不强制使用固定数量的特征点，则直接堆叠
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)
            descriptors = torch.stack(descriptors, 0)
        
        # 返回预测结果
        pred = {
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "descriptors": descriptors,
        }
        
        return pred

    def _aliked_distill_loss_from_keypoints(self, kpts_batch, aliked_keypoints_batch, img_shape, device):
        """
        Compute ALIKED distillation loss from pre-computed ALIKED keypoints.
        This is much faster than computing ALIKED on-the-fly.
        
        Args:
            kpts_batch: XFeat keypoint logits (B, C, H, W) where C=65
            aliked_keypoints_batch: Pre-computed ALIKED keypoints (B, N, 2) in image coordinates
            img_shape: (H, W) of the image
            device: Device for tensors
        
        Returns:
            losses: List of losses for each image
            accs: List of accuracies for each image
        """
        B, C, H_coarse, W_coarse = kpts_batch.shape  # H_coarse=H/8, W_coarse=W/8
        H_img, W_img = img_shape
        
        losses = []
        accs = []
        scale_factor = 8.0
        
        for b in range(B):
            aliked_kpts = aliked_keypoints_batch[b]  # (N, 2) in image coordinates
            
            # Convert to coarse resolution (H/8, W/8)
            aliked_kpts_coarse = aliked_kpts / scale_factor  # (N, 2)
            
            # Create labels: 64 bins for 8x8 subpixel grid
            labels = torch.ones((H_coarse, W_coarse), dtype=torch.long, device=device) * 64
            
            # Compute subpixel offsets
            aliked_kpts_int = aliked_kpts_coarse.long()
            offsets = ((aliked_kpts_coarse - aliked_kpts_int) * 8).long()  # (N, 2) in [0, 7]
            offsets_linear = offsets[:, 0] + 8 * offsets[:, 1]  # (N,)
            
            # Set labels at keypoint locations
            valid_mask = (aliked_kpts_int[:, 0] >= 0) & (aliked_kpts_int[:, 0] < W_coarse) & \
                        (aliked_kpts_int[:, 1] >= 0) & (aliked_kpts_int[:, 1] < H_coarse)
            valid_kpts = aliked_kpts_int[valid_mask]
            valid_offsets = offsets_linear[valid_mask]
            
            if len(valid_kpts) == 0:
                losses.append(torch.tensor(0.0, device=device))
                accs.append(torch.tensor(0.0, device=device))
                continue
            
            labels[valid_kpts[:, 1], valid_kpts[:, 0]] = valid_offsets
            
            # Compute loss
            kpts = kpts_batch[b].permute(1, 2, 0)  # (H, W, 65)
            kpts_flat = kpts.view(-1, C)  # (H*W, 65)
            labels_flat = labels.view(-1)  # (H*W,)
            
            # Sample positive and negative examples
            mask_pos = labels_flat < 64
            idxs_pos = mask_pos.nonzero().flatten()
            idxs_neg = (~mask_pos).nonzero().flatten()
            
            if len(idxs_pos) == 0:
                losses.append(torch.tensor(0.0, device=device))
                accs.append(torch.tensor(0.0, device=device))
                continue
            
            # Sample negatives (1/32 of positives)
            n_neg = max(1, len(idxs_pos) // 32)
            if len(idxs_neg) > n_neg:
                perm = torch.randperm(len(idxs_neg), device=device)[:n_neg]
                idxs_neg = idxs_neg[perm]
            
            idxs = torch.cat([idxs_pos, idxs_neg])
            kpts_sampled = kpts_flat[idxs]
            labels_sampled = labels_flat[idxs]
            
            # Compute accuracy
            with torch.no_grad():
                predicted = kpts_sampled.max(dim=-1)[1]
                acc = (labels_sampled == predicted).float().mean()
            
            # Compute loss
            kpts_log = F.log_softmax(kpts_sampled, dim=-1)
            loss = F.nll_loss(kpts_log, labels_sampled, reduction='mean')
            
            losses.append(loss)
            accs.append(acc)
        
        return losses, accs

    def loss(self, pred, data):
        """Compute XFeat training losses using ground truth correspondences."""
        if not self.conf.apply_xfeat_loss or self.dual_softmax_loss is None:
            return {"total": torch.tensor(0.0, device=next(self.net.parameters()).device)}, {}
        
        # Get images from both views
        image0 = data["view0"]["image"]  # (B, C, H, W)
        image1 = data["view1"]["image"]
        
        # Convert to grayscale if needed
        if image0.shape[1] == 3:
            p1 = image0.mean(1, keepdim=True)
            p2 = image1.mean(1, keepdim=True)
        else:
            p1, p2 = image0, image1
        
        # Forward pass through the model to get features, keypoints, and heatmaps
        feats1, kpts1, hmap1 = self.net(p1)
        feats2, kpts2, hmap2 = self.net(p2)
        
        # Get ground truth matches from pred (set by homography_matcher or depth_matcher)
        if "gt_matches0" not in pred or "gt_matches1" not in pred:
            # No ground truth available, return zero loss
            device = next(self.net.parameters()).device
            return {"total": torch.tensor(0.0, device=device)}, {}
        
        matches0 = pred["gt_matches0"]  # (B, N) - indices in image1, -1 = unmatched, -2 = ignore
        matches1 = pred["gt_matches1"]  # (B, M) - indices in image0
        
        # Get keypoints from predictions
        kp0 = pred.get("keypoints0")  # (B, N, 2) in original image coordinates
        kp1 = pred.get("keypoints1")  # (B, M, 2)
        
        if kp0 is None or kp1 is None:
            # Keypoints not available, return zero loss
            device = next(self.net.parameters()).device
            return {"total": torch.tensor(0.0, device=device)}, {}
        
        # Check if ALIKED keypoints are pre-computed and cached in data (best performance!)
        # This allows pre-computing ALIKED features offline and loading them from cache
        B = p1.shape[0]
        aliked_losses_p1 = None
        aliked_losses_p2 = None
        aliked_accs_p1 = None
        aliked_accs_p2 = None
        
        # Check for cached ALIKED keypoints in data (pre-computed offline)
        use_cached_aliked = False
        if self.use_aliked_distill:
            # Check if ALIKED keypoints are provided in view0/view1 cache
            # They come from dataloader as lists (one per batch item)
            view0_cache = data.get("view0", {})
            view1_cache = data.get("view1", {})
            
            # Handle both dict cache and list of caches (from dataloader collation)
            # ALIKED keypoints are stored as numpy arrays to avoid stacking issues
            if isinstance(view0_cache, dict) and "cache" in view0_cache:
                kpts0 = view0_cache["cache"].get("aliked_keypoints")
                kpts1 = view1_cache.get("cache", {}).get("aliked_keypoints")
                cached_aliked_kpts0_list = [kpts0] if kpts0 is not None else None
                cached_aliked_kpts1_list = [kpts1] if kpts1 is not None else None
            elif isinstance(view0_cache, list):
                # Batched: list of dicts, one per sample
                cached_aliked_kpts0_list = [v.get("cache", {}).get("aliked_keypoints") for v in view0_cache]
                cached_aliked_kpts1_list = [v.get("cache", {}).get("aliked_keypoints") for v in view1_cache]
            else:
                cached_aliked_kpts0_list = None
                cached_aliked_kpts1_list = None
            
            # Check if we have valid cached keypoints for all batch items
            if (cached_aliked_kpts0_list is not None and cached_aliked_kpts1_list is not None and
                len(cached_aliked_kpts0_list) == B and len(cached_aliked_kpts1_list) == B and
                all(k is not None for k in cached_aliked_kpts0_list) and
                all(k is not None for k in cached_aliked_kpts1_list)):
                
                use_cached_aliked = True
                # Process each batch item individually (variable-length keypoints)
                aliked_losses_p1 = []
                aliked_losses_p2 = []
                aliked_accs_p1 = []
                aliked_accs_p2 = []
                
                for b in range(B):
                    # Process single image
                    kpts1_single = kpts1[b:b+1]  # (1, C, H, W)
                    kpts2_single = kpts2[b:b+1]  # (1, C, H, W)
                    
                    # Convert numpy arrays to tensors if needed
                    kpts0_raw = cached_aliked_kpts0_list[b]
                    kpts1_raw = cached_aliked_kpts1_list[b]
                    
                    if isinstance(kpts0_raw, np.ndarray):
                        kpts0_single = torch.from_numpy(kpts0_raw).float().unsqueeze(0)  # (1, N, 2)
                    elif isinstance(kpts0_raw, torch.Tensor):
                        kpts0_single = kpts0_raw.unsqueeze(0) if kpts0_raw.dim() == 2 else kpts0_raw  # (1, N, 2)
                    else:
                        kpts0_single = None
                    
                    if isinstance(kpts1_raw, np.ndarray):
                        kpts1_single_cached = torch.from_numpy(kpts1_raw).float().unsqueeze(0)  # (1, N, 2)
                    elif isinstance(kpts1_raw, torch.Tensor):
                        kpts1_single_cached = kpts1_raw.unsqueeze(0) if kpts1_raw.dim() == 2 else kpts1_raw  # (1, N, 2)
                    else:
                        kpts1_single_cached = None
                    
                    if kpts0_single is not None and kpts1_single_cached is not None:
                        # Move to same device as input
                        kpts0_single = kpts0_single.to(p1.device)
                        kpts1_single_cached = kpts1_single_cached.to(p2.device)
                        
                        losses_p1, accs_p1 = self._aliked_distill_loss_from_keypoints(
                            kpts1_single, kpts0_single, p1.shape[-2:], p1.device
                        )
                        losses_p2, accs_p2 = self._aliked_distill_loss_from_keypoints(
                            kpts2_single, kpts1_single_cached, p2.shape[-2:], p2.device
                        )
                        
                        aliked_losses_p1.extend(losses_p1)
                        aliked_losses_p2.extend(losses_p2)
                        aliked_accs_p1.extend(accs_p1)
                        aliked_accs_p2.extend(accs_p2)
                    else:
                        # No valid keypoints for this batch item
                        aliked_losses_p1.append(torch.tensor(0.0, device=p1.device))
                        aliked_losses_p2.append(torch.tensor(0.0, device=p2.device))
                        aliked_accs_p1.append(torch.tensor(0.0, device=p1.device))
                        aliked_accs_p2.append(torch.tensor(0.0, device=p2.device))

        # Fallback: load ALIKED keypoints from an H5 file specified in the extractor conf.
        # This avoids requiring the dataset to pre-cache variable-length keypoints.
        if (
            self.use_aliked_distill
            and not use_cached_aliked
            and self._aliked_h5_path is not None
            and aliked_losses_p1 is None
            and ("name" in data)
        ):
            try:
                aliked_losses_p1 = []
                aliked_losses_p2 = []
                aliked_accs_p1 = []
                aliked_accs_p2 = []

                names = data["name"]
                view0 = data.get("view0", {})
                view1 = data.get("view1", {})
                for b in range(B):
                    name_b = names[b]
                    kpts_np = self._load_aliked_keypoints_for_name(name_b)
                    if kpts_np is None or len(kpts_np) == 0:
                        aliked_losses_p1.append(torch.tensor(0.0, device=p1.device))
                        aliked_losses_p2.append(torch.tensor(0.0, device=p2.device))
                        aliked_accs_p1.append(torch.tensor(0.0, device=p1.device))
                        aliked_accs_p2.append(torch.tensor(0.0, device=p2.device))
                        continue

                    kpts_orig = torch.from_numpy(kpts_np).to(device=p1.device, dtype=torch.float32)  # (N,2)
                    kpts_view0 = self._transform_aliked_keypoints_to_view(kpts_orig, view0, b, inverse=False)
                    kpts_view1 = self._transform_aliked_keypoints_to_view(kpts_orig.to(p2.device), view1, b, inverse=False)

                    kpts1_single = kpts1[b : b + 1]  # (1,C,H,W) logits for view0
                    kpts2_single = kpts2[b : b + 1]  # (1,C,H,W) logits for view1

                    losses_p1, accs_p1 = self._aliked_distill_loss_from_keypoints(
                        kpts1_single, kpts_view0[None], p1.shape[-2:], p1.device
                    )
                    losses_p2, accs_p2 = self._aliked_distill_loss_from_keypoints(
                        kpts2_single, kpts_view1[None], p2.shape[-2:], p2.device
                    )
                    aliked_losses_p1.append(losses_p1[0])
                    aliked_losses_p2.append(losses_p2[0])
                    aliked_accs_p1.append(accs_p1[0])
                    aliked_accs_p2.append(accs_p2[0])
            except Exception:
                # If anything goes wrong reading/transforming, just disable the fallback for this batch.
                aliked_losses_p1 = None
                aliked_losses_p2 = None
                aliked_accs_p1 = None
                aliked_accs_p2 = None
        
        # Note: ALIKED features must be pre-computed and cached in dataset
        # If cached features are not available, distillation loss will be zero
        
        # Compute losses for each batch item
        loss_ds_list = []
        loss_coords_list = []
        loss_kp_list = []
        loss_kp_pos_list = []
        metrics = {}
        
        h_coarse, w_coarse = feats1.shape[-2], feats1.shape[-1]  # (H/8, W/8)
        
        for b in range(B):
            # Extract positive matches (ignore -1 unmatched and -2 ignore)
            valid0 = (matches0[b] >= 0)  # Positive matches only
            if valid0.sum() < 30:  # Skip if too few correspondences
                continue
            
            idx0 = torch.arange(len(matches0[b]), device=p1.device)[valid0]
            idx1 = matches0[b][valid0].long()
            
            # Get keypoint coordinates at original resolution
            pts1_orig = kp0[b][valid0]  # (N, 2) in original image coordinates
            pts2_orig = kp1[b][idx1]     # (N, 2) in original image coordinates
            
            # Scale to coarse resolution (H/8, W/8)
            # XFeat outputs features at 1/8 resolution
            scale_factor = 8.0
            pts1_coarse = pts1_orig / scale_factor  # (N, 2) [x, y]
            pts2_coarse = pts2_orig / scale_factor   # (N, 2) [x, y]
            
            # Extract features at corresponding locations (convert to integer indices)
            # Clamp coordinates to valid range: x in [0, w_coarse-1], y in [0, h_coarse-1]
            pts1_int = pts1_coarse.long()
            pts2_int = pts2_coarse.long()
            
            # Ensure indices are within bounds (x, y separately)
            pts1_int[:, 0] = pts1_int[:, 0].clamp(0, w_coarse - 1)  # x coordinate
            pts1_int[:, 1] = pts1_int[:, 1].clamp(0, h_coarse - 1)  # y coordinate
            pts2_int[:, 0] = pts2_int[:, 0].clamp(0, w_coarse - 1)  # x coordinate
            pts2_int[:, 1] = pts2_int[:, 1].clamp(0, h_coarse - 1)  # y coordinate
            
            m1 = feats1[b, :, pts1_int[:, 1], pts1_int[:, 0]].permute(1, 0)  # (N, 64)
            m2 = feats2[b, :, pts2_int[:, 1], pts2_int[:, 0]].permute(1, 0)  # (N, 64)
            
            # Extract heatmaps
            h1 = hmap1[b, 0, pts1_int[:, 1], pts1_int[:, 0]]
            h2 = hmap2[b, 0, pts2_int[:, 1], pts2_int[:, 0]]
            
            # Fine matcher for subpixel refinement
            coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))  # (N, 64)
            
            # Compute losses
            loss_ds, conf = self.dual_softmax_loss(m1, m2)
            loss_coords, acc_coords = self.coordinate_classification_loss(
                coords1, pts1_coarse, pts2_coarse, conf
            )
            
            # Keypoint position distillation loss (ALIKED from precomputed keypoints only)
            if aliked_losses_p1 is not None:
                # Use pre-computed ALIKED losses from cached features
                loss_kp_pos1 = aliked_losses_p1[b]
                loss_kp_pos2 = aliked_losses_p2[b]
                acc_pos1 = aliked_accs_p1[b]
                acc_pos2 = aliked_accs_p2[b]
                loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
            else:
                # Skip distillation loss if precomputed ALIKED keypoints are not available
                loss_kp_pos = torch.tensor(0.0, device=p1.device)
                acc_pos1 = torch.tensor(0.0, device=p1.device)
                acc_pos2 = torch.tensor(0.0, device=p1.device)
            
            # Keypoint reliability loss
            loss_kp = self.keypoint_loss(h1, conf) + self.keypoint_loss(h2, conf)
            
            # Collect losses
            loss_ds_list.append(loss_ds)
            loss_coords_list.append(loss_coords)
            loss_kp_list.append(loss_kp)
            loss_kp_pos_list.append(loss_kp_pos)
            
            # Metrics (only from first batch item for logging)
            if b == 0:
                acc_coarse = self.check_accuracy(m1, m2)
                # Convert scalar metrics to 1D tensors for AverageMetric compatibility
                device = p1.device
                # Helper function to convert scalar to 1D tensor
                def to_1d_tensor(val):
                    if isinstance(val, torch.Tensor):
                        if val.dim() == 0:
                            return val.unsqueeze(0)
                        elif val.dim() == 1:
                            return val
                        else:
                            return val.flatten()
                    else:
                        return torch.tensor([float(val)], device=device)
                
                metrics["xfeat/acc_coarse"] = to_1d_tensor(acc_coarse)
                metrics["xfeat/acc_fine"] = to_1d_tensor(acc_coords)
                if aliked_losses_p1 is not None:
                    acc_kp_pos_val = (acc_pos1 + acc_pos2) / 2.0
                    metrics["xfeat/acc_kp_pos"] = to_1d_tensor(acc_kp_pos_val)
        
        if len(loss_ds_list) == 0:
            device = next(self.net.parameters()).device
            # Return 1D tensor for AverageMetric compatibility
            return {"total": torch.tensor([0.0], device=device)}, {}
        
        # Aggregate losses across batch
        loss_ds_mean = torch.stack(loss_ds_list).mean()
        loss_coords_mean = torch.stack(loss_coords_list).mean()
        loss_kp_mean = torch.stack(loss_kp_list).mean()
        loss_kp_pos_mean = torch.stack(loss_kp_pos_list).mean()
        
        total_loss = (loss_ds_mean + loss_coords_mean + loss_kp_mean + loss_kp_pos_mean) * self.conf.xfeat_loss_weight
        
        # Ensure all losses are 1D tensors for AverageMetric compatibility
        # .mean() returns 0D tensors, so we need to unsqueeze them
        def ensure_1d(t):
            return t.unsqueeze(0) if t.dim() == 0 else t
        
        losses = {
            "xfeat/dual_softmax": ensure_1d(loss_ds_mean),
            "xfeat/coordinate": ensure_1d(loss_coords_mean),
            "xfeat/keypoint": ensure_1d(loss_kp_mean),
            "xfeat/keypoint_pos": ensure_1d(loss_kp_pos_mean),
            "total": ensure_1d(total_loss),
        }
        
        return losses, metrics


# For testing the XFeat extractor directly
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Test the XFeat extractor
    print("Testing XFeat extractor...")
    
    # Create model
    model = XFeat({})
    
    # Load a test image if available, otherwise create a random one
    try:
        # Try to load an image from accelerated_features/assets
        assets_path = Path(accelerated_features_path) / "assets"
        img_path = None
        
        if assets_path.exists():
            for file in assets_path.glob("*.jpg"):
                img_path = file
                break
            if img_path is None:
                for file in assets_path.glob("*.png"):
                    img_path = file
                    break
        
        if img_path is not None and img_path.exists():
            print(f"Loading test image from {img_path}")
            img = Image.open(img_path).convert('RGB')
            img = np.array(img) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
            
            # 创建一个小批次进行测试
            batch_size = 2
            img = img.repeat(batch_size, 1, 1, 1)
        else:
            print("No test image found, creating random image")
            batch_size = 4
            img = torch.rand(batch_size, 3, 480, 640)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using random image instead")
        batch_size = 4
        img = torch.rand(batch_size, 3, 480, 640)
    
    # Run inference
    data = {"image": img}
    with torch.no_grad():
        pred = model(data)
    
    # Print results
    print(f"Batch size: {batch_size}")
    for i in range(batch_size):
        print(f"Image {i}: Extracted {len(pred['keypoints'][i])} keypoints")
    print(f"Keypoints shape: {pred['keypoints'].shape}")
    print(f"Descriptors shape: {pred['descriptors'].shape}")
    
    print("XFeat extractor test completed successfully!") 