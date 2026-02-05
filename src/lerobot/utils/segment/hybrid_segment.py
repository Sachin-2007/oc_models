"""
Hybrid Segmentation Pipeline

Uses the best approach for each object:
- TIGER: HSV color detection → SAM bounding box (reliable, 100% detection)
- ELEPHANT: GroundingDINO → SAM bounding box (better for grey objects)

Models are loaded once and reused for all images.
"""

import os
import logging
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from transformers import (
    SamModel, SamProcessor,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)
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
OUTPUT_DIR = Path("hybrid_segment_results")

# HSV color ranges for tiger (orange) - works reliably
ORANGE_LOWER = np.array([8, 150, 120])
ORANGE_UPPER = np.array([22, 255, 255])

# GroundingDINO prompt for elephant
ELEPHANT_PROMPT = "grey elephant toy."


def find_tiger_box(image_np, min_area=500, expand_ratio=0.5):
    """
    Find tiger using HSV color detection.
    
    Returns: [x1, y1, x2, y2] bounding box or None
    """
    h, w = image_np.shape[:2]
    
    # Convert to HSV
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Create mask for orange
    mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    
    # Get bounding box and expand
    x, y, bw, bh = cv2.boundingRect(largest)
    expand_w = int(bw * expand_ratio / 2)
    expand_h = int(bh * expand_ratio / 2)
    
    x1 = max(0, x - expand_w)
    y1 = max(0, y - expand_h)
    x2 = min(w, x + bw + expand_w)
    y2 = min(h, y + bh + expand_h)
    
    return [x1, y1, x2, y2]


def find_elephant_box(image_pil, gd_processor, gd_model):
    """
    Find elephant using GroundingDINO.
    
    Returns: [x1, y1, x2, y2] bounding box or None
    """
    inputs = gd_processor(images=image_pil, text=ELEPHANT_PROMPT, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = gd_model(**inputs)
    
    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image_pil.size[::-1]]
    )[0]
    
    if len(results["boxes"]) == 0:
        return None
    
    # Get highest scoring box
    max_idx = results["scores"].argmax()
    box = results["boxes"][max_idx].tolist()
    
    return box


def segment_with_sam(image_pil, box, sam_processor, sam_model):
    """
    Segment using SAM with a bounding box prompt.
    
    Returns: Binary mask as numpy array (H, W)
    """
    if box is None:
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
    
    input_boxes = [[box]]
    inputs = sam_processor(image_pil, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )[0]
    
    # Combine masks
    h, w = image_pil.height, image_pil.width
    combined = np.zeros((h, w), dtype=bool)
    for m in masks:
        combined = np.logical_or(combined, m[0].cpu().numpy())
    
    return (combined * 255).astype(np.uint8)


def draw_overlay(image_pil, tiger_mask, elephant_mask):
    """Create overlay visualization."""
    vis_img = image_pil.copy().convert("RGBA")
    h, w = image_pil.height, image_pil.width
    
    # Tiger mask overlay (orange)
    if tiger_mask is not None and tiger_mask.max() > 0:
        tiger_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        tiger_overlay[tiger_mask > 127] = [255, 165, 0, 120]
        tiger_img = Image.fromarray(tiger_overlay)
        vis_img = Image.alpha_composite(vis_img, tiger_img)
    
    # Elephant mask overlay (blue)
    if elephant_mask is not None and elephant_mask.max() > 0:
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
    
    # 3. Load ALL models once
    logger.info(f"Loading models on {DEVICE}...")
    
    # SAM
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # GroundingDINO
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE)
    
    logger.info("Models loaded!")
    
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
    
    # 5. PASS 1: Color detection for tiger (fast)
    logger.info("Pass 1: Color detection for tiger...")
    all_tiger_boxes = []
    for i, img_np in enumerate(first_frames):
        box = find_tiger_box(img_np)
        all_tiger_boxes.append(box)
    
    tiger_detected = sum(1 for b in all_tiger_boxes if b is not None)
    logger.info(f"  Tiger detected: {tiger_detected}/{len(first_frames)}")
    
    # 6. PASS 2: GroundingDINO for elephant
    logger.info("Pass 2: GroundingDINO for elephant...")
    all_elephant_boxes = []
    for i, img_np in enumerate(first_frames):
        image_pil = Image.fromarray(img_np)
        box = find_elephant_box(image_pil, gd_processor, gd_model)
        all_elephant_boxes.append(box)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed: {i+1}/{len(first_frames)}")
    
    elephant_detected = sum(1 for b in all_elephant_boxes if b is not None)
    logger.info(f"  Elephant detected: {elephant_detected}/{len(first_frames)}")
    
    # 7. PASS 3: SAM segmentation for all
    logger.info("Pass 3: SAM segmentation...")
    all_tiger_masks = []
    all_elephant_masks = []
    
    for i, img_np in enumerate(first_frames):
        image_pil = Image.fromarray(img_np)
        
        tiger_mask = segment_with_sam(image_pil, all_tiger_boxes[i], sam_processor, sam_model)
        elephant_mask = segment_with_sam(image_pil, all_elephant_boxes[i], sam_processor, sam_model)
        
        all_tiger_masks.append(tiger_mask)
        all_elephant_masks.append(elephant_mask)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  SAM: {i+1}/{len(first_frames)}")
    
    # 8. Save results
    logger.info(f"Saving results to {OUTPUT_DIR}...")
    for i in range(len(first_frames)):
        Image.fromarray(all_tiger_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_mask_tiger.png")
        Image.fromarray(all_elephant_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_mask_elephant.png")
        
        overlay = draw_overlay(Image.fromarray(first_frames[i]), all_tiger_masks[i], all_elephant_masks[i])
        overlay.save(OUTPUT_DIR / f"ep_{i:02d}_overlay.png")
    
    # Save detected boxes visualization for first few
    for i in range(min(5, len(first_frames))):
        vis_img = Image.fromarray(first_frames[i]).copy()
        draw = ImageDraw.Draw(vis_img)
        if all_tiger_boxes[i]:
            draw.rectangle(all_tiger_boxes[i], outline="orange", width=3)
        if all_elephant_boxes[i]:
            draw.rectangle(all_elephant_boxes[i], outline="blue", width=3)
        vis_img.save(OUTPUT_DIR / f"ep_{i:02d}_boxes.png")
    
    logger.info(f"Done! Tiger: {tiger_detected}/50, Elephant: {elephant_detected}/50")


if __name__ == "__main__":
    main()
