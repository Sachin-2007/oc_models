import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("segment_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting script...")

try:
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from transformers import SamModel, SamProcessor
    logger.info("Imports successful.")
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)


# Constants
DATASET_NAME = "aadarshram/pick_place_tape"
OUTPUT_DIR = "d:/dev/segment/segmented_frames"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODES_TO_PROCESS = 50

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    logger.info(f"Loading models on {DEVICE}...")
    
    # Grounding DINO for detection
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE)
    
    # SAM for segmentation
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    return gd_processor, gd_model, sam_processor, sam_model

def get_bounding_box(image_pil, text_prompt, processor, model):
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    target_sizes = torch.tensor([image_pil.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.25,
        target_sizes=target_sizes
    )[0]
    
    return results["boxes"], results["scores"], results["labels"]

def segment_with_sam(image_pil, boxes, processor, model):
    if len(boxes) == 0:
        return None
        
    # SAM expects input_boxes as a list of list of boxes: [[x1, y1, x2, y2], ...]
    # And input_points/labels can be provided.
    # Transformers SAM processor usage:
    # inputs = processor(image, input_boxes=[[boxes]], return_tensors="pt")
    
    # We take the box with highest score from Grounding DINO usually, or all of them.
    # Let's take specific boxes.
    
    # boxes is tensor (N, 4)
    input_boxes = [boxes.tolist()] # SAM processor expects a list of boxes for the image
    
    inputs = processor(image_pil, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process masks
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, 
        inputs["original_sizes"], 
        inputs["reshaped_input_sizes"]
    )[0]
    
    return masks

def main():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        # Fallback for older versions or if structure is different
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            logger.error("LeRobotDataset not found in lerobot.datasets or lerobot.common.datasets")
            return

    # Load dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = LeRobotDataset(DATASET_NAME, root=".")
    
    # Load models
    gd_processor, gd_model, sam_processor, sam_model = load_models()
    
    logger.info(f"Dataset has {ds.num_episodes} episodes. Processing first {EPISODES_TO_PROCESS}...")
    
    for episode_idx in range(ds.num_episodes):
        if episode_idx >= EPISODES_TO_PROCESS:
            break
            
        logger.info(f"Processing Episode {episode_idx}")
        
        if hasattr(ds, 'episode_data_index'):
             from_idx = ds.episode_data_index["from"][episode_idx].item()
        else:
             # Fallback for newer lerobot where meta.episodes is a HF Dataset
             from_idx = ds.meta.episodes[episode_idx]["dataset_from_index"]
        
        # Get item
        item = ds[from_idx]
        
        # Image key: 'observation.images.top_phone'
        if 'observation.images.top_phone' not in item:
            logger.warning(f"Image key not found in episode {episode_idx}")
            continue
            
        img_tensor = item['observation.images.top_phone']
        # LeRobot: C, H, W in float or uint8.
        # Check shape/dtype
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy() # C,H,W -> H,W,C
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = np.array(img_tensor)
            
        image_pil = Image.fromarray(img_np)
        
        # 1. Detect "black tape" vs "robotic arm" to avoid confusion
        # We include "robotic arm" so the model can assign it to that class instead of "black tape"
        boxes, scores, labels = get_bounding_box(image_pil, "black tape . robotic arm .", gd_processor, gd_model)
        
        # Filter for "black tape" only
        target_indices = [i for i, label in enumerate(labels) if "tape" in label]
        
        if len(target_indices) > 0:
            boxes = boxes[target_indices]
            scores = scores[target_indices]
            labels = [labels[i] for i in target_indices]
            
            logger.info(f"  Found {len(boxes)} matches for 'black tape' (after filtering robot arm)")
            
            # 2. Segment
            masks = segment_with_sam(image_pil, boxes, sam_processor, sam_model)
            
            # 3. Visualize
            # Convert to OpenCV for drawing
            vis_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Overlay masks
            # masks is shape (num_boxes, num_masks_per_box, H, W)
            # Usually num_masks_per_box is 3 (multimask output). We take best one?
            # Or simplified: (num_boxes, 1, H, W) output from simple inference.
            
            # Let's flatten masks
            if masks is not None:
                combined_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
                for i in range(len(masks)):
                    # taking the first mask hypothesis for each box
                    m = masks[i][0].cpu().numpy() 
                    combined_mask = np.logical_or(combined_mask, m)
                
                # Draw mask overlay
                colored_mask = np.zeros_like(vis_img)
                colored_mask[combined_mask] = [0, 255, 0] # Green
                
                vis_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)
                
                # Draw boxes - REMOVED as per request
                # for box in boxes:
                #     x1, y1, x2, y2 = map(int, box.tolist())
                #     cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Save
            filename = os.path.join(OUTPUT_DIR, f"episode_{episode_idx}_tape.png")
            cv2.imwrite(filename, vis_img)
            logger.info(f"  Saved {filename}")
        else:
            logger.info("  No tape detected.")
            # Save original anyway?
            # Requirement: "segment the black tape". If none, maybe skip.

if __name__ == "__main__":
    main()
