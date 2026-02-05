"""
Color-Based Point Prompting Segmentation Pipeline

Uses HSV color thresholding to find candidate points for SAM segmentation,
replacing GroundingDINO text prompting which sometimes fails to detect the elephant.

Workflow:
1. Load all first frames from dataset
2. Run HSV color detection on ALL images (batch), cache point candidates
3. Load SAM model once
4. Run SAM inference on all images using cached points
5. Save visualization outputs
"""

import os
import logging
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from transformers import SamModel, SamProcessor
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
OUTPUT_DIR = Path("color_segment_results")

# HSV color ranges (H: 0-179, S: 0-255, V: 0-255 in OpenCV)
# Orange for tiger - more saturated to exclude wood grain
ORANGE_LOWER = np.array([8, 150, 120])
ORANGE_UPPER = np.array([22, 255, 255])

# Grey for elephant - mid-grey, exclude dark robot arm and light background
GREY_LOWER = np.array([0, 0, 60])
GREY_UPPER = np.array([180, 50, 130])


def find_color_points(image_np, lower_hsv, upper_hsv, min_area=500, top_k=1, exclude_robot=False):
    """
    Find centroids of color regions in HSV space.
    
    Args:
        image_np: RGB image as numpy array (H, W, 3)
        lower_hsv: Lower HSV bound as numpy array
        upper_hsv: Upper HSV bound as numpy array
        min_area: Minimum contour area to consider
        top_k: Number of largest contours to return (default 1 for single object)
        exclude_robot: If True, mask out bottom-center region where robot arm is
        
    Returns:
        List of (x, y) centroid points, color mask
    """
    h, w = image_np.shape[:2]
    
    # Convert RGB to BGR for OpenCV, then to HSV
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Create mask for color range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Exclude robot arm region (bottom 35%, center 40% of width - focused on robot only)
    if exclude_robot:
        robot_top = int(h * 0.65)  # Start from 65% down
        robot_left = int(w * 0.30)  # Start from 30% from left
        robot_right = int(w * 0.70)  # End at 70% from left
        mask[robot_top:, robot_left:robot_right] = 0
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and sort by size (largest first)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append((area, contour))
    
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    
    # Take only top_k largest contours
    points = []
    for area, contour in valid_contours[:top_k]:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    
    return points, mask


def find_color_boxes(image_np, lower_hsv, upper_hsv, min_area=500, top_k=1, exclude_robot=False, expand_ratio=0.5):
    """
    Find bounding boxes of color regions in HSV space.
    
    Args:
        image_np: RGB image as numpy array (H, W, 3)
        lower_hsv: Lower HSV bound as numpy array
        upper_hsv: Upper HSV bound as numpy array
        min_area: Minimum contour area to consider
        top_k: Number of largest contours to return
        exclude_robot: If True, mask out robot arm region
        expand_ratio: Expand box by this ratio (0.5 = 50% larger)
        
    Returns:
        List of [x1, y1, x2, y2] bounding boxes, color mask
    """
    h, w = image_np.shape[:2]
    
    # Convert RGB to BGR for OpenCV, then to HSV
    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Create mask for color range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Exclude robot arm region
    if exclude_robot:
        robot_top = int(h * 0.65)
        robot_left = int(w * 0.30)
        robot_right = int(w * 0.70)
        mask[robot_top:, robot_left:robot_right] = 0
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and sort by size (largest first)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append((area, contour))
    
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    
    # Get bounding boxes for top_k largest contours
    boxes = []
    for area, contour in valid_contours[:top_k]:
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Expand box by expand_ratio
        expand_w = int(bw * expand_ratio / 2)
        expand_h = int(bh * expand_ratio / 2)
        
        x1 = max(0, x - expand_w)
        y1 = max(0, y - expand_h)
        x2 = min(w, x + bw + expand_w)
        y2 = min(h, y + bh + expand_h)
        
        boxes.append([x1, y1, x2, y2])
    
    return boxes, mask


def load_sam_model():
    """Load SAM model and processor once."""
    logger.info(f"Loading SAM model on {DEVICE}...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    return sam_processor, sam_model


def segment_with_sam_points(image_pil, points, processor, model):
    """
    Segment image using SAM with point prompts.
    
    Args:
        image_pil: PIL Image
        points: List of (x, y) tuples
        processor: SAM processor
        model: SAM model
        
    Returns:
        Combined binary mask as numpy array (H, W)
    """
    if not points:
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
    
    # SAM expects input_points as [[[x1, y1], [x2, y2], ...]] for batch of 1
    # and input_labels as [[1, 1, ...]] where 1 = foreground
    input_points = [[list(p) for p in points]]
    input_labels = [[1] * len(points)]
    
    inputs = processor(
        image_pil, 
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )[0]
    
    # Combine all masks - masks shape: (num_points, num_mask_predictions, H, W)
    # We take the first (best) mask prediction for each point
    h, w = image_pil.height, image_pil.width
    combined = np.zeros((h, w), dtype=bool)
    
    for m in masks:
        # m is (num_predictions, H, W), take first prediction
        combined = np.logical_or(combined, m[0].cpu().numpy())
    
    return (combined * 255).astype(np.uint8)


def segment_with_sam_boxes(image_pil, boxes, processor, model):
    """
    Segment image using SAM with bounding box prompts.
    
    Args:
        image_pil: PIL Image
        boxes: List of [x1, y1, x2, y2] bounding boxes
        processor: SAM processor
        model: SAM model
        
    Returns:
        Combined binary mask as numpy array (H, W)
    """
    if not boxes:
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
    
    # SAM expects input_boxes as [[[x1, y1, x2, y2], ...]] for batch of 1
    input_boxes = [boxes]
    
    inputs = processor(
        image_pil, 
        input_boxes=input_boxes,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )[0]
    
    # Combine all masks
    h, w = image_pil.height, image_pil.width
    combined = np.zeros((h, w), dtype=bool)
    
    for m in masks:
        # m is (num_predictions, H, W), take first prediction
        combined = np.logical_or(combined, m[0].cpu().numpy())
    
    return (combined * 255).astype(np.uint8)


def draw_points_on_image(image_pil, orange_points, grey_points):
    """Draw detected points on image for visualization."""
    vis_img = image_pil.copy()
    draw = ImageDraw.Draw(vis_img)
    
    # Draw orange points (tiger)
    for x, y in orange_points:
        draw.ellipse([x-8, y-8, x+8, y+8], fill="orange", outline="black", width=2)
    
    # Draw grey points (elephant)
    for x, y in grey_points:
        draw.ellipse([x-8, y-8, x+8, y+8], fill="blue", outline="black", width=2)
    
    return vis_img


def draw_result_overlay(image_pil, mask_tiger, mask_elephant):
    """Create overlay visualization with both masks."""
    vis_img = image_pil.copy().convert("RGBA")
    h, w = image_pil.height, image_pil.width
    
    # Tiger mask overlay (orange)
    if mask_tiger is not None:
        tiger_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        tiger_mask = mask_tiger > 127
        tiger_overlay[tiger_mask] = [255, 165, 0, 100]  # Orange with alpha
        tiger_img = Image.fromarray(tiger_overlay, mode="RGBA")
        vis_img = Image.alpha_composite(vis_img, tiger_img)
    
    # Elephant mask overlay (blue)
    if mask_elephant is not None:
        elephant_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        elephant_mask = mask_elephant > 127
        elephant_overlay[elephant_mask] = [0, 100, 255, 100]  # Blue with alpha
        elephant_img = Image.fromarray(elephant_overlay, mode="RGBA")
        vis_img = Image.alpha_composite(vis_img, elephant_img)
    
    return vis_img


def main():
    # 1. Setup output directory
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    
    # 2. Load dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = LeRobotDataset(DATASET_NAME, root=".")
    
    # 3. Collect first frames from each episode
    logger.info("Collecting first frame of each episode...")
    first_frames = []
    for ep_idx in range(ds.num_episodes):
        ep_meta = ds.meta.episodes[ep_idx]
        global_idx = ep_meta["dataset_from_index"]
        item = ds[global_idx]
        img_tensor = item[IMAGE_KEY]
        
        # Tensor -> Numpy
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
    
    # 4. PASS 1: Color detection on ALL images (cache bounding boxes)
    logger.info("Pass 1: Running color detection on all images...")
    all_orange_boxes = []
    all_grey_boxes = []
    all_orange_masks = []
    all_grey_masks = []
    
    for i, img_np in enumerate(first_frames):
        orange_boxes, orange_mask = find_color_boxes(img_np, ORANGE_LOWER, ORANGE_UPPER)
        grey_boxes, grey_mask = find_color_boxes(img_np, GREY_LOWER, GREY_UPPER, exclude_robot=True)
        
        all_orange_boxes.append(orange_boxes)
        all_grey_boxes.append(grey_boxes)
        all_orange_masks.append(orange_mask)
        all_grey_masks.append(grey_mask)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Color detection: {i+1}/{len(first_frames)}")
    
    logger.info(f"Color detection complete. Saving intermediate results...")
    
    # Save color detection debug images for first few frames
    for i in range(min(5, len(first_frames))):
        Image.fromarray(all_orange_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_color_orange.png")
        Image.fromarray(all_grey_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_color_grey.png")
        
        # Draw bounding boxes visualization
        vis_img = Image.fromarray(first_frames[i]).copy()
        draw = ImageDraw.Draw(vis_img)
        for box in all_orange_boxes[i]:
            draw.rectangle(box, outline="orange", width=3)
        for box in all_grey_boxes[i]:
            draw.rectangle(box, outline="blue", width=3)
        vis_img.save(OUTPUT_DIR / f"ep_{i:02d}_detected_boxes.png")
    
    # 5. Load SAM model once
    sam_processor, sam_model = load_sam_model()
    
    # 6. PASS 2: SAM segmentation on ALL images using bounding boxes
    logger.info("Pass 2: Running SAM segmentation on all images (using bounding boxes)...")
    all_tiger_masks = []
    all_elephant_masks = []
    
    for i, img_np in enumerate(first_frames):
        image_pil = Image.fromarray(img_np)
        
        # Segment tiger (orange boxes)
        tiger_mask = segment_with_sam_boxes(
            image_pil, all_orange_boxes[i], sam_processor, sam_model
        )
        
        # Segment elephant (grey boxes)
        elephant_mask = segment_with_sam_boxes(
            image_pil, all_grey_boxes[i], sam_processor, sam_model
        )
        
        all_tiger_masks.append(tiger_mask)
        all_elephant_masks.append(elephant_mask)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  SAM segmentation: {i+1}/{len(first_frames)}")
    
    # 7. Save all results
    logger.info(f"Saving {len(all_tiger_masks)} mask pairs to {OUTPUT_DIR}...")
    
    for i in range(len(first_frames)):
        # Individual masks
        Image.fromarray(all_tiger_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_mask_tiger.png")
        Image.fromarray(all_elephant_masks[i]).save(OUTPUT_DIR / f"ep_{i:02d}_mask_elephant.png")
        
        # Overlay visualization
        overlay = draw_result_overlay(
            Image.fromarray(first_frames[i]),
            all_tiger_masks[i],
            all_elephant_masks[i]
        )
        overlay.save(OUTPUT_DIR / f"ep_{i:02d}_overlay.png")
    
    logger.info(f"Done! Results saved to {OUTPUT_DIR}/")
    
    # Print summary
    detected_tiger = sum(1 for boxes in all_orange_boxes if boxes)
    detected_elephant = sum(1 for boxes in all_grey_boxes if boxes)
    logger.info(f"Summary: Detected tiger in {detected_tiger}/{len(first_frames)} frames")
    logger.info(f"Summary: Detected elephant in {detected_elephant}/{len(first_frames)} frames")


if __name__ == "__main__":
    main()
