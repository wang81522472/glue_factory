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
else:
    # 当作为模块导入时使用相对导入
    from ..base_model import BaseModel
    from ..utils.misc import pad_to_length, pad_and_stack

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
        "use_aliked_distill": True,  # Use ALIKED for distillation (requires pre-computed ALIKED features in dataset cache)
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        # Ensure accelerated_features is available
        _ensure_accelerated_features()
        
        # Import training utilities (only if trainable)
        if conf.trainable:
            try:
                if accelerated_features_path not in sys.path:
                    sys.path.append(accelerated_features_path)
                
                # Import loss functions and utilities
                # Note: The losses module imports ALike at the top level, which may fail
                # We'll handle this by importing the module and extracting functions manually
                try:
                    # Try importing utils first (doesn't depend on ALike)
                    try:
                        from accelerated_features.modules.training.utils import check_accuracy
                    except (ImportError, ModuleNotFoundError):
                        from modules.training.utils import check_accuracy
                    
                    # Import the losses module, handling ALike import failure
                    # We'll import the module and extract the functions we need
                    losses_module = None
                    try:
                        try:
                            import accelerated_features.modules.training.losses as losses_module
                        except (ImportError, ModuleNotFoundError):
                            import modules.training.losses as losses_module
                    except (ImportError, ModuleNotFoundError) as e:
                        error_msg = str(e).lower()
                        if 'alike' in error_msg or 'extract_alike_kpts' in error_msg:
                            # ALike import failed, but we can still use the other loss functions
                            # by importing them directly from the source
                            print(f"Warning: ALike import failed, but core loss functions are still available.")
                            print("Attempting to import core loss functions directly...")
                            
                            # Import core loss functions that don't depend on ALike
                            import importlib.util
                            import os
                            
                            # Find the losses.py file
                            losses_path = None
                            # Try both possible paths
                            potential_paths = [
                                os.path.join(accelerated_features_path, 'modules', 'training', 'losses.py'),
                                os.path.join(accelerated_features_path, 'accelerated_features', 'modules', 'training', 'losses.py'),
                            ]
                            for potential_path in potential_paths:
                                if os.path.exists(potential_path):
                                    losses_path = potential_path
                                    break
                            
                            if losses_path and os.path.exists(losses_path):
                                # Read and execute the module, but skip the ALike import
                                with open(losses_path, 'r') as f:
                                    losses_code = f.read()
                                
                                # Replace the problematic import with a try-except or skip entirely if not trainable
                                if conf.trainable:
                                    # Only try to import ALike if trainable
                                    losses_code = losses_code.replace(
                                        'from third_party.alike_wrapper import extract_alike_kpts',
                                        '''try:
    from third_party.alike_wrapper import extract_alike_kpts
    ALIKE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    extract_alike_kpts = None
    ALIKE_AVAILABLE = False'''
                                    )
                                else:
                                    # Skip ALike import entirely when trainable is False
                                    losses_code = losses_code.replace(
                                        'from third_party.alike_wrapper import extract_alike_kpts',
                                        '''# ALike import skipped: trainable is False
extract_alike_kpts = None
ALIKE_AVAILABLE = False'''
                                    )
                                
                                # Execute the modified code
                                losses_namespace = {}
                                exec(compile(losses_code, losses_path, 'exec'), losses_namespace)
                                losses_module = type(sys)('losses_module')
                                losses_module.__dict__.update(losses_namespace)
                            else:
                                raise ImportError(f"Could not find losses.py at expected paths")
                        else:
                            raise
                    
                    # Extract the functions we need
                    dual_softmax_loss = losses_module.dual_softmax_loss
                    coordinate_classification_loss = losses_module.coordinate_classification_loss
                    keypoint_loss = losses_module.keypoint_loss
                    
                    # Try to get alike_distill_loss if available and trainable
                    if conf.trainable:
                        self.alike_distill_loss = getattr(losses_module, 'alike_distill_loss', None)
                        self.use_alike_loss = getattr(losses_module, 'ALIKE_AVAILABLE', False)
                        
                        if self.alike_distill_loss is None or not self.use_alike_loss:
                            print("Note: ALike distillation loss is not available. Using ALIKED for distillation if enabled.")
                            self.alike_distill_loss = None
                            self.use_alike_loss = False
                    else:
                        # Skip ALike initialization when trainable is False
                        print("Skipping ALike initialization: trainable is False")
                        self.alike_distill_loss = None
                        self.use_alike_loss = False
                    
                    # Store loss functions
                    self.dual_softmax_loss = dual_softmax_loss
                    self.coordinate_classification_loss = coordinate_classification_loss
                    self.keypoint_loss = keypoint_loss
                    self.check_accuracy = check_accuracy
                    
                except Exception as e:
                    # If import still fails, try a simpler approach: define the functions locally
                    print(f"Warning: Could not import loss functions from module: {e}")
                    print("Attempting to define core loss functions locally...")
                    
                    # Define the core loss functions locally (they don't depend on ALike)
                    def dual_softmax_loss(X, Y, temp=0.2):
                        if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
                            raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')
                        dist_mat = (X @ Y.t()) * temp
                        conf_matrix12 = F.log_softmax(dist_mat, dim=1)
                        conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)
                        with torch.no_grad():
                            conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
                            conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
                            conf = conf12 * conf21
                        target = torch.arange(len(X), device=X.device)
                        loss = F.nll_loss(conf_matrix12, target) + F.nll_loss(conf_matrix21, target)
                        return loss, conf
                    
                    def coordinate_classification_loss(coords1, pts1, pts2, conf):
                        with torch.no_grad():
                            coords1_detached = pts1 * 8
                            offsets1_detached = (coords1_detached/8) - (coords1_detached/8).long()
                            offsets1_detached = (offsets1_detached * 8).long()
                            labels1 = offsets1_detached[:, 0] + 8*offsets1_detached[:, 1]
                        coords1_log = F.log_softmax(coords1, dim=-1)
                        predicted = coords1.max(dim=-1)[1]
                        acc = (labels1 == predicted)
                        acc = acc[conf > 0.1]
                        acc = acc.sum() / len(acc) if len(acc) > 0 else torch.tensor(0.0, device=coords1.device)
                        loss = F.nll_loss(coords1_log, labels1, reduction='none')
                        conf = conf / conf.sum()
                        loss = (loss * conf).sum()
                        return loss * 2., acc
                    
                    def keypoint_loss(heatmap, target):
                        L1_loss = F.l1_loss(heatmap, target)
                        return L1_loss * 3.0
                    
                    # Import check_accuracy
                    try:
                        try:
                            from accelerated_features.modules.training.utils import check_accuracy
                        except (ImportError, ModuleNotFoundError):
                            from modules.training.utils import check_accuracy
                    except Exception:
                        # Define check_accuracy locally if import fails
                        def check_accuracy(X, Y, pts1=None, pts2=None, plot=False):
                            with torch.no_grad():
                                dist_mat = X @ Y.t()
                                nn = torch.argmax(dist_mat, dim=1)
                                correct = nn == torch.arange(len(X), device=X.device)
                                acc = correct.sum().item() / len(X)
                                return acc
                    
                    self.dual_softmax_loss = dual_softmax_loss
                    self.coordinate_classification_loss = coordinate_classification_loss
                    self.keypoint_loss = keypoint_loss
                    self.check_accuracy = check_accuracy
                    self.alike_distill_loss = None
                    self.use_alike_loss = False
                    print("Successfully loaded core loss functions (ALike distillation disabled).")
                    
            except Exception as e:
                # If training modules are not available, disable training
                print(f"Warning: Could not import XFeat training modules: {e}")
                print("XFeat training will be disabled.")
                self.dual_softmax_loss = None
                self.coordinate_classification_loss = None
                self.keypoint_loss = None
                self.alike_distill_loss = None
                self.use_alike_loss = False
                self.check_accuracy = None
        else:
            # Skip all training-related imports when trainable is False
            print("Skipping training utilities import: trainable is False")
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
        if not conf.trainable:
            for p in self.net.parameters():
                p.requires_grad = False
        
        # ALIKED distillation uses pre-computed features from dataset cache
        # No need to initialize ALIKED model here - features are loaded from disk
        self.use_aliked_distill = conf.get("use_aliked_distill", True)
        
        # Move to GPU by default for better performance
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xfeat.to(self.device)

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
            
            # Keypoint position distillation loss (ALIKED from cache or ALike fallback)
            if self.use_aliked_distill and aliked_losses_p1 is not None:
                # Use pre-computed ALIKED losses from cached features
                loss_kp_pos1 = aliked_losses_p1[b]
                loss_kp_pos2 = aliked_losses_p2[b]
                acc_pos1 = aliked_accs_p1[b]
                acc_pos2 = aliked_accs_p2[b]
                loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
            elif self.use_alike_loss and self.alike_distill_loss is not None:
                # Fall back to ALike if available (still sequential, but less common)
                loss_kp_pos1, acc_pos1 = self.alike_distill_loss(kpts1[b], p1[b])
                loss_kp_pos2, acc_pos2 = self.alike_distill_loss(kpts2[b], p2[b])
                loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
            else:
                # Skip distillation loss if neither is available
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
                if (self.use_aliked_distill and aliked_losses_p1 is not None) or \
                   (self.use_alike_loss and self.alike_distill_loss is not None):
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