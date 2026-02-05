
import os
import sys
import logging

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_NAME = "aadarshram/pick_place_tiger_near_elephant"
OUTPUT_REPO_ID = "aadarshram/pick_place_tiger_near_elephant_segmented"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_KEY = "observation.images.top_phone" # Assuming this is the main view

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
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=target_sizes
    )[0]
    
    return results["boxes"], results["scores"], results["labels"]

def segment_with_sam(image_pil, boxes, processor, model):
    if len(boxes) == 0:
        return None
        
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
    # Load dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = LeRobotDataset(DATASET_NAME, root=".")
    
    # Load models
    gd_processor, gd_model, sam_processor, sam_model = load_models()
    
    # We will create a new dataset by copying the old one structures and adding new features
    # But since LeRobotDataset is immutable-ish (parquet based), we might want to just iterate and save to a new location.
    # However, to be efficient, we can probably just buffer the new data and write it.
    
    # Actually, the easiest way to add columns to a LeRobotDataset is to re-write it.
    # We can use the logic from `lerobot_edit_dataset.py` or just build a new dataset manually.
    # Since we are adding 2 video-like features (masks), we should probably treat them as videos or compressed images.
    # Masks are binary (or uint8 0/255).
    
    # Let's create a temporary directory for the new dataset
    output_dir = Path("data") / OUTPUT_REPO_ID
    if output_dir.exists():
        shutil.rmtree(output_dir)
    # output_dir.mkdir(parents=True) -> Removed to let LeRobotDataset.create handle it
    
    logger.info(f"Processing {ds.num_episodes} episodes...")
    
    # We need to collect all frames for the new columns
    # To avoid huge memory usage, we process episode by episode and write to disk?
    # LeRobotDataset usually writes chunks.
    
    # Let's try to just generate the masks as temporary video files or image folders, 
    # and then attach them?
    # Or simpler: Just iterate, modify the dict, and use a writer.
    
    # We'll use a simple loop to process and collect data, then create a new LeRobotDataset.
    # Be careful with RAM.
    
    # Define new features
    features = ds.features.copy()
    features["observation.images.i_mask"] = {
        "dtype": "image",
        "shape": (1, 96, 96), # Assuming resized? Or original? Original is safer.
        "names": ["channel", "height", "width"]
    }
    features["observation.images.f_mask"] = {
        "dtype": "image",
        "shape": (1, 96, 96), # We will check the actual image size
        "names": ["channel", "height", "width"]
    }
    
    # Start loop
    new_episodes_data = [] # Buffer for metadata if needed, but we might just use the writer directly if we constructed a custom writer.
    
    # But LeRobotDataset doesn't expose a simple "append column" API.
    # We can use `LeRobotDataset.create` to start a new one, and then `save_episode`.
    
    # First, get image size from first frame to set features correctly
    item = ds[0]
    img_tensor = item[IMAGE_KEY]
    c, h, w = img_tensor.shape
    features["observation.images.i_mask"]["shape"] = (1, h, w)
    features["observation.images.f_mask"]["shape"] = (1, h, w)
    features[IMAGE_KEY]["shape"] = (c, h, w) # Ensure original is correct
    
    # Create new dataset writer
    new_ds_meta = LeRobotDataset.create(
        repo_id=OUTPUT_REPO_ID,
        fps=ds.fps,
        features=features,
        robot_type=ds.meta.robot_type,
        root=output_dir,
        use_videos=True # We want to save masks as videos probably? Or images? 
                        # If we say 'image' dtype, LeRobot saves as video if use_videos=True.
                        # Ideally masks are compressed losslessly or effectively.
                        # mp4 h264 is lossy. 
                        # Maybe 'hevc' or 'libsvtav1'?
    )
    
    for episode_idx in range(ds.num_episodes):
        logger.info(f"Processing Episode {episode_idx}/{ds.num_episodes}")
        
        # Get episode data
        ep_meta = ds.meta.episodes[episode_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        num_frames = to_idx - from_idx
        
        # Load all frames for this episode
        # We can iterate or load batch. Batch might be faster for GPU inference.
        
        i_masks_list = []
        f_masks_list = []
        
        # We need to preserve all other data
        episode_data = {k: [] for k in ds.features}
        
        for frame_idx in range(from_idx, to_idx):
            item = ds[frame_idx]
            
            # Collect existing data
            for k in ds.features:
                val = item[k]
                if isinstance(val, torch.Tensor):
                    val = val.numpy() # Convert to numpy for storage/processing
                episode_data[k].append(val)
                
            # Process Image for Masks
            img_tensor = item[IMAGE_KEY] # C, H, W
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            else:
                 img_np = np.array(img_tensor)
                 
            image_pil = Image.fromarray(img_np)
            
            # --- Detect Tiger (i_mask) ---
            # "orange tiger"
            boxes, scores, labels = get_bounding_box(image_pil, "orange tiger", gd_processor, gd_model)
            # Filter?
            i_mask = np.zeros((h, w), dtype=np.uint8)
            if len(boxes) > 0:
                # Take best score
                # best_idx = scores.argmax()
                # boxes = boxes[best_idx:best_idx+1]
                # Segment
                masks = segment_with_sam(image_pil, boxes, sam_processor, sam_model)
                if masks is not None:
                     # masks: (num_boxes, 1, H, W)
                     # Combine all masks found
                     combined = np.zeros((h, w), dtype=bool)
                     for m in masks:
                         combined = np.logical_or(combined, m[0].cpu().numpy())
                     i_mask[combined] = 255
            
            # --- Detect Elephant (f_mask) ---
            # "grey elephant"
            boxes, scores, labels = get_bounding_box(image_pil, "grey elephant", gd_processor, gd_model)
            f_mask = np.zeros((h, w), dtype=np.uint8)
            if len(boxes) > 0:
                 masks = segment_with_sam(image_pil, boxes, sam_processor, sam_model)
                 if masks is not None:
                     combined = np.zeros((h, w), dtype=bool)
                     for m in masks:
                         combined = np.logical_or(combined, m[0].cpu().numpy())
                     f_mask[combined] = 255
            
            i_masks_list.append(i_mask[None, ...]) # C, H, W -> 1, H, W
            f_masks_list.append(f_mask[None, ...])

        # Convert lists to arrays
        for k in episode_data:
            episode_data[k] = np.stack(episode_data[k])
            
        episode_data["observation.images.i_mask"] = np.stack(i_masks_list)
        episode_data["observation.images.f_mask"] = np.stack(f_masks_list)
        
        # Save episode
        new_ds_meta.meta.save_episode(
            episode_index=episode_idx,
            episode_length=num_frames,
            episode_tasks=ds.meta.episodes[episode_idx]["tasks"], # Re-use tasks
            episode_stats={}, # Stats will be computed? LeRobotDataset usually computes stats on the fly or we need to pass them.
                              # Actually `save_episode` calls `aggregate_stats`. We should pass stats for this episode.
                              # But to calculate stats properly we need to compute them.
                              # `compute_episode_stats` helper exists.
            episode_metadata={}
        )
        
        # LeRobotDataset separates metadata saving from video saving?
        # `save_episode` buffers metadata.
        # But where do the videos/images go? 
        # `save_episode` doesn't seem to write video files in the snippets I saw. 
        # Ah, looking at `LeRobotDataset` code (or `lerobot_record.py`), usually there is an `AsyncImageWriter` or similar.
        # Or `lerobot_edit_dataset` uses transformers. 
        
        # Modification: `LeRobotDataset` is a read class mostly, or write via `save_episode` for metadata.
        # But for video data, we need to encode it.
        # In `lerobot_record.py` it uses `ImageWriter`.
        # Since I am in a script, I should manually encode the videos.
        
        # Encode videos for this episode
        # We need to write 3 videos: top_phone, i_mask, f_mask (plus any others present)
        # Note: existing videos (top_phone) might already exist. We can copy them or re-encode.
        # To be safe and clean, re-encoding is easier logic, though slower.
        
        # Wait, if I use `save_episode`, it expects the files to be in the right place?
        # `_save_episode_metadata` doesn't move files.
        
        # I need to write the video files to `videos/{key}/chunk-{c}/file-{i}.mp4`
        # `new_ds_meta.get_video_file_path(episode_idx, key)` gives the path.
        
        # Let's save the videos.
        for key in features:
            if features[key]["dtype"] == "video": # or image if use_videos=True?
                # Check if it is a video key
                # In `configuration_oc_diffusion`, "observation.images" are image features.
                # In dataset, if dtype is image and use_videos is True, it is stored as video.
                
                # Get data
                vid_data = episode_data[key] # T, C, H, W
                # Expects T, H, W, C for encoding typically? Or T, C, H, W?
                # `encode_video_frames` takes a directory of images or similar? 
                # Let's check `lerobot.datasets.video_utils`. 
                # `encode_video_frames(img_dir, temp_path, fps, ...)`
                
                # Use `write_video` equivalent?
                # `torchvision.io.write_video` takes T, H, W, C.
                
                # Convert to T, H, W, C
                vid_data_perm = np.transpose(vid_data, (0, 2, 3, 1))
                
                # If 1 channel (mask), broadcast to 3? Or is 1 channel supported?
                # Most codecs prefer 3 channels.
                if vid_data_perm.shape[-1] == 1:
                    vid_data_perm = np.repeat(vid_data_perm, 3, axis=-1)
                
                # Path
                vid_path = new_ds_meta.get_video_file_path(episode_idx, key)
                vid_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write
                torchvision.io.write_video(
                    str(vid_path), 
                    torch.from_numpy(vid_data_perm), 
                    fps=ds.fps,
                    video_codec="libx264", # Use h264 for compatibility and speed, masks should be fine.
                    options={"crf": "18"} # Low CRF for quality
                )

    # Finalize
    new_ds_meta._close_writer()
    logger.info("Done!")

if __name__ == "__main__":
    import torchvision
    main()
