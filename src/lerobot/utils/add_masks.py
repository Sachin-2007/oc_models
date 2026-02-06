"""
Add segmentation masks to a LeRobot dataset.

Uses hybrid segmentation approach from segment/hybrid_segment.py:
- TIGER (i_mask): HSV color detection → SAM
- ELEPHANT (f_mask): GroundingDINO → SAM
"""

import argparse
import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Import segmentation functions from hybrid_segment
from lerobot.utils.segment.hybrid_segment import (
    find_tiger_box,
    find_elephant_box,
    segment_with_sam,
    DEVICE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_models(device):
    """Load GroundingDINO and SAM models."""
    logger.info(f"Loading models on {device}...")
    
    # Grounding DINO for elephant detection
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
    
    # SAM for segmentation
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    return gd_processor, gd_model, sam_processor, sam_model


def main():
    parser = argparse.ArgumentParser(description="Add segmentation masks to a LeRobot dataset using hybrid approach.")
    parser.add_argument("--repo-id", type=str, required=True, help="Source LeRobot dataset repository ID.")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Target LeRobot dataset repository ID.")
    parser.add_argument("--image-key", type=str, default="observation.images.top_phone", help="Key of the image feature to segment.")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--push-to-hub", action="store_true", help="Push the result to Hugging Face Hub.")

    args = parser.parse_args()
    
    device = args.device
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading source dataset: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id, root=".")
    
    # Load models
    gd_processor, gd_model, sam_processor, sam_model = load_models(device)
    
    # Prepare output directory
    output_dir = Path("data") / args.output_repo_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    # Define new features
    features = ds.features.copy()
    
    # Update image key if default not found and only one image exists
    img_keys = [k for k in features if k.startswith("observation.images")]
    if args.image_key not in features:
        if len(img_keys) > 0:
            logger.warning(f"Image key '{args.image_key}' not found. Using '{img_keys[0]}' instead.")
            args.image_key = img_keys[0]
        else:
            raise ValueError(f"No image features found in dataset. features: {features.keys()}")

    # Determine shape from first frame
    item = ds[0]
    img_tensor = item[args.image_key]
    c, h, w = img_tensor.shape
    
    # Add mask features for tiger (i_mask) and elephant (f_mask)
    # Using 3 channels for video compatibility
    features["observation.images.i_mask"] = {
        "dtype": "video", 
        "shape": (3, h, w),
        "names": ["channel", "height", "width"]
    }
    features["observation.images.f_mask"] = {
        "dtype": "video", 
        "shape": (3, h, w),
        "names": ["channel", "height", "width"]
    }
    
    logger.info(f"Creating new dataset: {args.output_repo_id} with i_mask (tiger) and f_mask (elephant)")
    
    new_ds = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        fps=ds.fps,
        features=features,
        robot_type=ds.meta.robot_type,
        root=output_dir,
        use_videos=True 
    )

    for episode_idx in range(ds.num_episodes):
        logger.info(f"Processing Episode {episode_idx + 1}/{ds.num_episodes}")
        
        ep_meta = ds.meta.episodes[episode_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        num_frames = to_idx - from_idx
        
        # 1. Process First Frame for Masks using Hybrid Approach
        first_item = ds[from_idx]
        img_tensor = first_item[args.image_key]
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = np.array(img_tensor)
            
        image_pil = Image.fromarray(img_np)
        
        # Tiger: HSV color detection -> SAM (from hybrid_segment)
        tiger_box = find_tiger_box(img_np)
        tiger_mask = segment_with_sam(image_pil, tiger_box, sam_processor, sam_model)
        
        # Elephant: GroundingDINO -> SAM (from hybrid_segment)
        elephant_box = find_elephant_box(image_pil, gd_processor, gd_model)
        elephant_mask = segment_with_sam(image_pil, elephant_box, sam_processor, sam_model)
        
        # Convert to 3-channel for video compatibility (H, W) -> (3, H, W)
        tiger_mask_3ch = np.stack([tiger_mask, tiger_mask, tiger_mask], axis=0)
        elephant_mask_3ch = np.stack([elephant_mask, elephant_mask, elephant_mask], axis=0)
        
        # Get task from source dataset
        task_list = ds.meta.episodes[episode_idx].get("tasks", [])
        if task_list is None or len(task_list) == 0:
            task_str = "pick and place"
        else:
            task_str = task_list[0]
        
        # Add frames using add_frame API
        # Keys that are auto-generated by add_frame and should not be included
        auto_keys = {'index', 'episode_index', 'frame_index', 'timestamp', 'task_index'}
        
        for frame_idx in range(from_idx, to_idx):
            item = ds[frame_idx]
            
            # Build frame dict with data features only
            frame = {"task": task_str}
            
            for k in ds.features:
                # Skip auto-generated keys
                if k in auto_keys:
                    continue
                    
                val = item[k]
                if isinstance(val, torch.Tensor):
                    val = val.numpy()
                
                # Convert channel-first (C,H,W) to channel-last (H,W,C) for images/videos
                if ds.features[k]["dtype"] in ["image", "video"] and val.ndim == 3:
                    val = np.transpose(val, (1, 2, 0))  # C,H,W -> H,W,C
                    
                frame[k] = val
            
            # Add masks (repeated from first frame), also in channel-last
            frame["observation.images.i_mask"] = np.transpose(tiger_mask_3ch, (1, 2, 0))
            frame["observation.images.f_mask"] = np.transpose(elephant_mask_3ch, (1, 2, 0))
            
            new_ds.add_frame(frame)
        
        # Save episode
        new_ds.save_episode()

    new_ds.finalize()
    logger.info("Dataset creation complete.")
    
    if args.push_to_hub:
        logger.info(f"Pushing to hub: {args.output_repo_id}")
        new_ds.push_to_hub()
        logger.info("Push complete.")


if __name__ == "__main__":
    main()
