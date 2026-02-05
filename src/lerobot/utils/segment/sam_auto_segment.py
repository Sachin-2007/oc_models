"""
Alternative Segmentation Pipeline: SAM Auto Masks + Color Classification

Instead of using color detection to find prompts, this approach:
1. Uses SAM's automatic mask generator to find ALL objects in the image
2. Classifies each mask by its dominant color (orange=tiger, grey=elephant)
3. Selects the best mask for each category

This is more robust as SAM finds objects first, then we classify them.
"""

import os
import logging
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from transformers import SamModel, SamProcessor, pipeline
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_NAME = "aadarshram/pick_place_tiger_near_elephant"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_KEY = "observation.images.top_phone"
OUTPUT_DIR = Path("sam_auto_segment_results")


def get_mask_dominant_color(image_np, mask):
    """
    Get the dominant HSV color within a mask region.
    
    Returns: (hue, saturation, value) median values
    """
    # Apply mask to image
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return None
    
    # Convert to HSV
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Get pixels within mask
    h_values = hsv[:, :, 0][mask_bool]
    s_values = hsv[:, :, 1][mask_bool]
    v_values = hsv[:, :, 2][mask_bool]
    
    return np.median(h_values), np.median(s_values), np.median(v_values)


def classify_mask(image_np, mask):
    """
    Classify a mask as 'tiger', 'elephant', 'robot', or 'background'.
    
    Returns: (class_name, confidence_score)
    """
    color = get_mask_dominant_color(image_np, mask)
    if color is None:
        return "background", 0.0
    
    h, s, v = color
    
    # Orange tiger: H=5-25, high saturation
    if 5 <= h <= 25 and s >= 80 and v >= 80:
        return "tiger", s / 255.0  # Higher saturation = more confident
    
    # Grey elephant: Low saturation, wide value range
    # Elephant is distinctly grey - low saturation is key
    if s < 100 and 40 < v < 180:
        # Not too dark (robot) and not too bright (background)
        return "elephant", (100 - s) / 100.0  # Lower saturation = more confident
    
    # Dark (robot or shadow): Very low value
    if v < 40:
        return "robot", 0.0
    
    return "background", 0.0


def segment_with_sam_auto(image_pil, generator):
    """
    Use SAM to automatically segment all objects in the image.
    
    Returns: List of masks with their predictions
    """
    outputs = generator(image_pil, points_per_batch=64)
    return outputs


def draw_overlay(image_pil, tiger_mask, elephant_mask):
    """Create overlay visualization."""
    vis_img = image_pil.copy().convert("RGBA")
    h, w = image_pil.height, image_pil.width
    
    # Tiger mask overlay (orange)
    if tiger_mask is not None:
        tiger_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        tiger_overlay[tiger_mask > 127] = [255, 165, 0, 120]
        tiger_img = Image.fromarray(tiger_overlay)
        vis_img = Image.alpha_composite(vis_img, tiger_img)
    
    # Elephant mask overlay (blue)  
    if elephant_mask is not None:
        elephant_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        elephant_overlay[elephant_mask > 127] = [0, 100, 255, 120]
        elephant_img = Image.fromarray(elephant_overlay)
        vis_img = Image.alpha_composite(vis_img, elephant_img)
    
    return vis_img


def main():
    # 1. Setup
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    
    # 2. Load dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = LeRobotDataset(DATASET_NAME, root=".")
    
    # 3. Load SAM automatic mask generator
    logger.info(f"Loading SAM mask generator on {DEVICE}...")
    generator = pipeline(
        "mask-generation",
        model="facebook/sam-vit-base",
        device=0 if DEVICE == "cuda" else -1
    )
    
    # 4. Collect first frames
    logger.info("Collecting first frame of each episode...")
    first_frames = []
    for ep_idx in range(ds.num_episodes):
        ep_meta = ds.meta.episodes[ep_idx]
        global_idx = ep_meta["dataset_from_index"]
        item = ds[global_idx]
        img_tensor = item[IMAGE_KEY]
        
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.numpy()
        else:
            img_np = np.array(img_tensor)
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        first_frames.append(img_np)
    
    logger.info(f"Collected {len(first_frames)} frames.")
    
    # 5. Process each frame
    all_tiger_masks = []
    all_elephant_masks = []
    
    for i, img_np in enumerate(first_frames):
        image_pil = Image.fromarray(img_np)
        
        # Generate all masks
        outputs = generator(image_pil, points_per_batch=64)
        
        # Classify each mask
        tiger_candidates = []  # (mask, score)
        elephant_candidates = []
        
        for out in outputs["masks"]:
            mask_np = np.array(out)
            mask_pixels = mask_np.sum()
            total_pixels = mask_np.shape[0] * mask_np.shape[1]
            
            # Skip very small masks (less than 2% of image) or very large (background)
            mask_ratio = mask_pixels / total_pixels
            if mask_ratio < 0.02 or mask_ratio > 0.5:
                continue
            
            class_name, score = classify_mask(img_np, mask_np)
            
            # Use both color score and mask size for ranking
            combined_score = score * (mask_pixels ** 0.5)  # Prefer larger masks
            
            if class_name == "tiger" and score > 0:
                tiger_candidates.append((mask_np, combined_score))
            elif class_name == "elephant" and score > 0:
                elephant_candidates.append((mask_np, combined_score))
        
        # Select best mask for each
        tiger_mask = None
        elephant_mask = None
        
        if tiger_candidates:
            # Sort by score and size
            tiger_candidates.sort(key=lambda x: (x[1], x[0].sum()), reverse=True)
            tiger_mask = (tiger_candidates[0][0] * 255).astype(np.uint8)
        
        if elephant_candidates:
            elephant_candidates.sort(key=lambda x: (x[1], x[0].sum()), reverse=True)
            elephant_mask = (elephant_candidates[0][0] * 255).astype(np.uint8)
        
        all_tiger_masks.append(tiger_mask if tiger_mask is not None else np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8))
        all_elephant_masks.append(elephant_mask if elephant_mask is not None else np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8))
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Processed: {i+1}/{len(first_frames)}")
    
    # 6. Save results
    logger.info(f"Saving results to {OUTPUT_DIR}...")
    for i in range(len(first_frames)):
        Image.fromarray(all_tiger_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_mask_tiger.png")
        Image.fromarray(all_elephant_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_mask_elephant.png")
        
        overlay = draw_overlay(Image.fromarray(first_frames[i]), all_tiger_masks[i], all_elephant_masks[i])
        overlay.save(OUTPUT_DIR / f"ep_{i:02d}_overlay.png")
    
    # Summary
    detected_tiger = sum(1 for m in all_tiger_masks if m.max() > 0)
    detected_elephant = sum(1 for m in all_elephant_masks if m.max() > 0)
    logger.info(f"Done! Tiger: {detected_tiger}/50, Elephant: {detected_elephant}/50")


if __name__ == "__main__":
    main()
