"""
Template Matching Segmentation Pipeline

Uses template matching to find objects:
1. Extract tiger and elephant templates from a reference first frame
2. Use template matching to find the objects in all frames
3. Expand matched regions and use SAM for precise segmentation

This works well for consistent-looking objects across frames.
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
OUTPUT_DIR = Path("template_segment_results")

# Template regions from first frame (manually identified good bounds)
# These will be extracted from the first frame as templates
# Format: [x1, y1, x2, y2] in the reference frame
TIGER_TEMPLATE_REGION = [335, 125, 460, 260]   # Approximate tiger location in frame 0
ELEPHANT_TEMPLATE_REGION = [140, 145, 270, 290]  # Approximate elephant location in frame 0


def extract_template(image_np, region):
    """Extract template from image."""
    x1, y1, x2, y2 = region
    return image_np[y1:y2, x1:x2].copy()


def find_template(image_np, template, threshold=0.5, expand_ratio=0.3):
    """
    Find template in image using normalized cross-correlation.
    
    Returns: [x1, y1, x2, y2] bounding box or None
    """
    h, w = image_np.shape[:2]
    th, tw = template.shape[:2]
    
    # Convert to grayscale for matching
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    templ_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    
    # Template matching using normalized cross-correlation
    result = cv2.matchTemplate(img_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    
    # Find best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val < threshold:
        return None, max_val
    
    # Get bounding box
    x, y = max_loc
    bw, bh = tw, th
    
    # Expand box
    expand_w = int(bw * expand_ratio / 2)
    expand_h = int(bh * expand_ratio / 2)
    
    x1 = max(0, x - expand_w)
    y1 = max(0, y - expand_h)
    x2 = min(w, x + bw + expand_w)
    y2 = min(h, y + bh + expand_h)
    
    return [x1, y1, x2, y2], max_val


def segment_with_sam(image_pil, box, sam_processor, sam_model):
    """Segment using SAM with bounding box prompt."""
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
    
    h, w = image_pil.height, image_pil.width
    combined = np.zeros((h, w), dtype=bool)
    for m in masks:
        combined = np.logical_or(combined, m[0].cpu().numpy())
    
    return (combined * 255).astype(np.uint8)


def draw_overlay(image_pil, tiger_mask, elephant_mask):
    """Create overlay visualization."""
    vis_img = image_pil.copy().convert("RGBA")
    h, w = image_pil.height, image_pil.width
    
    if tiger_mask is not None and tiger_mask.max() > 0:
        tiger_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        tiger_overlay[tiger_mask > 127] = [255, 165, 0, 120]
        tiger_img = Image.fromarray(tiger_overlay)
        vis_img = Image.alpha_composite(vis_img, tiger_img)
    
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
    
    # 3. Load SAM model
    logger.info(f"Loading SAM model on {DEVICE}...")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
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
    
    # 5. Extract templates from first frame (episode 0)
    logger.info("Extracting templates from reference frame...")
    ref_frame = first_frames[0]
    tiger_template = extract_template(ref_frame, TIGER_TEMPLATE_REGION)
    elephant_template = extract_template(ref_frame, ELEPHANT_TEMPLATE_REGION)
    
    # Save templates for visualization
    Image.fromarray(tiger_template).save(OUTPUT_DIR / "template_tiger.png")
    Image.fromarray(elephant_template).save(OUTPUT_DIR / "template_elephant.png")
    
    # 6. Find templates in all frames
    logger.info("Finding templates in all frames...")
    all_tiger_boxes = []
    all_elephant_boxes = []
    tiger_scores = []
    elephant_scores = []
    
    for i, img_np in enumerate(first_frames):
        tiger_box, t_score = find_template(img_np, tiger_template, threshold=0.3)
        elephant_box, e_score = find_template(img_np, elephant_template, threshold=0.3)
        
        all_tiger_boxes.append(tiger_box)
        all_elephant_boxes.append(elephant_box)
        tiger_scores.append(t_score)
        elephant_scores.append(e_score)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Template matching: {i+1}/{len(first_frames)}")
    
    tiger_detected = sum(1 for b in all_tiger_boxes if b is not None)
    elephant_detected = sum(1 for b in all_elephant_boxes if b is not None)
    logger.info(f"Template matching: Tiger {tiger_detected}/50, Elephant {elephant_detected}/50")
    
    # 7. SAM segmentation
    logger.info("SAM segmentation...")
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
    
    # Save detected boxes and scores for first few
    for i in range(min(5, len(first_frames))):
        vis_img = Image.fromarray(first_frames[i]).copy()
        draw = ImageDraw.Draw(vis_img)
        if all_tiger_boxes[i]:
            draw.rectangle(all_tiger_boxes[i], outline="orange", width=3)
            draw.text((all_tiger_boxes[i][0], all_tiger_boxes[i][1]-15), f"t:{tiger_scores[i]:.2f}", fill="orange")
        if all_elephant_boxes[i]:
            draw.rectangle(all_elephant_boxes[i], outline="blue", width=3)
            draw.text((all_elephant_boxes[i][0], all_elephant_boxes[i][1]-15), f"e:{elephant_scores[i]:.2f}", fill="blue")
        vis_img.save(OUTPUT_DIR / f"ep_{i:02d}_boxes.png")
    
    logger.info(f"Done! Tiger: {tiger_detected}/50, Elephant: {elephant_detected}/50")
    logger.info(f"Avg tiger score: {np.mean(tiger_scores):.3f}, Avg elephant score: {np.mean(elephant_scores):.3f}")


if __name__ == "__main__":
    main()
