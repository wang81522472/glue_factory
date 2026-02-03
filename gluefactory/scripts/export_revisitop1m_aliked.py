import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf

from gluefactory.models import get_model
from gluefactory.settings import DATA_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.utils.tools import get_device
from gluefactory.utils.image import ImagePreprocessor, load_image
from gluefactory import logger

class Revisitop1MDataset(torch.utils.data.Dataset):
    def __init__(self, root, list_file=None, preprocessing=None):
        self.root = Path(root)
        if list_file and Path(list_file).exists():
            with open(list_file, "r") as f:
                self.images = f.read().splitlines()
            logging.info(f"Loaded {len(self.images)} images from {list_file}")
        else:
            # Glob
            logging.info(f"Globbing images in {self.root}...")
            # revisitop1m typically has jpg folder. We glob everything.
            self.images = sorted(list(self.root.glob("**/*.jpg")))
            self.images = [str(p.relative_to(self.root)) for p in self.images]
            logging.info(f"Found {len(self.images)} images.")
        
        self.preprocessor = ImagePreprocessor(preprocessing or {})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        path = self.root / name
        
        # load_image returns tensor (C, H, W) in [0,1]
        try:
            img = load_image(path)
        except Exception as e:
            logging.warning(f"Failed to load image {path}: {e}")
            # Return dummy
            img = torch.zeros(3, 1024, 1024)
            
        data = {"name": name, "image": img}
        data = {**data, **self.preprocessor(data["image"])}
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=str(DATA_PATH / "revisitop1m"))
    parser.add_argument("--export_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_root = Path(args.dataset_path)
    if not data_root.exists():
        logging.error(f"Dataset path {data_root} does not exist.")
        return

    # ALIKED config
    # Copied/Adapted from export_local_features.py
    # We use 1600 resize by default for ALIKED export as per common practice in this repo
    # To support batching, we must use fixed number of keypoints (force_num_keypoints=True)
    # otherwise ALIKED model fails to stack variable number of keypoints.
    resize = 1600
    model_conf = {
        "name": "extractors.aliked",
        "model_name": "aliked-n16rot",
        "pretrained": True,
        "detection_threshold": -1, # Set to -1 to enable top-k mode
        "max_num_keypoints": 8000,
        "force_num_keypoints": True,
        "nms_radius": 2,
    }

    if args.export_name is None:
        args.export_name = f"r{resize}_aliked-n16rot-k8000"

    # Initialize Dataset
    list_file = data_root / "revisitop1m.txt"
    dataset = Revisitop1MDataset(
        data_root, 
        list_file=list_file if list_file.exists() else None,
        preprocessing={"resize": resize, "square_pad": True}
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    conf = OmegaConf.create({"model": model_conf})
    
    # Output file
    feature_file = DATA_PATH / "exports" / "revisitop1m" / (args.export_name + ".h5")
    feature_file.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"Exporting features to {feature_file}")
    
    device = torch.device(args.device)
    model = get_model(conf.model.name)(conf.model).eval().to(device)

    # Keys to export
    # keys = ["keypoints", "descriptors", "keypoint_scores"]
    keys = ["keypoints"]

    export_predictions(loader, model, feature_file, as_half=True, keys=keys)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
