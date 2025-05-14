"""XFeat feature extractor integration for glue-factory.
This module integrates the XFeat feature extractor from accelerated_features into the glue-factory framework.

Authors: XFeat - https://github.com/verlab/accelerated_features
Integration: glue-factory team
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

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

# 查找并添加accelerated_features路径
accelerated_features_path = find_accelerated_features()
if accelerated_features_path not in sys.path:
    sys.path.append(accelerated_features_path)

# Import XFeat from accelerated_features
try:
    from modules.xfeat import XFeat as OriginalXFeat
    from modules.interpolator import InterpolateSparse2d
except ImportError:
    raise ImportError(
        "Cannot import XFeat from accelerated_features. "
        "Make sure the accelerated_features repository is available "
        "at the path: " + accelerated_features_path
    )


class XFeat(BaseModel):
    default_conf = {
        "weights": None,  # Path to weights file, None for default
        "max_num_keypoints": 4096,  # Maximum number of keypoints (for glue-factory consistency)
        "top_k": None,  # Maximum number of keypoints (will use max_num_keypoints if None)
        "detection_threshold": 0.05,  # Detection threshold
        "force_num_keypoints": False,  # Force exact number of keypoints
    }

    required_data_keys = ["image"]

    def _init(self, conf):
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
        
        # Move to GPU by default for better performance
        self.device = torch.device('cuda')
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

    def loss(self, pred, data):
        raise NotImplementedError


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