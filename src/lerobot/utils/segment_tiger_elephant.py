
import os
import sys
import logging
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil
from pathlib import Path
import torchvision
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_NAME = "aadarshram/pick_place_tiger_near_elephant"
OUTPUT_REPO_ID = "RaspberryVitriol/oc_segment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_KEY = "observation.images.top_phone"
PROMPT_I = "orange tiger."
PROMPT_F = "grey toy."
BATCH_SIZE = 1 # Conservative batch size

# Output directory for mask visualization
VIZ_DIR = Path("final_masks_viz")

def load_models():
    logger.info(f"Loading models on {DEVICE}...")
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE)
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    return gd_processor, gd_model, sam_processor, sam_model

def get_bounding_boxes_batch(image_pil_list, text_prompt, processor, model):
    # GD Processor can handle list of images? 
    # Usually yes, but text might need to be repeated or just one string?
    # Checking HuggingFace docs: text (str, List[str], List[List[str]])
    # If list of images and single string, it might broadcast? 
    # Safe bet: Repeat prompt list.
    
    texts = [text_prompt] * len(image_pil_list)
    inputs = processor(images=image_pil_list, text=texts, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results_list = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=[img.size[::-1] for img in image_pil_list]
    )
    
    # Filter: Keep only Top-1 detection
    for res in results_list:
        if res["scores"].numel() > 1:
            max_idx = res["scores"].argmax()
            res["scores"] = res["scores"][max_idx:max_idx+1]
            res["boxes"] = res["boxes"][max_idx:max_idx+1]
            res["labels"] = res["labels"][max_idx:max_idx+1]
            
    return results_list

def segment_with_sam_batch(image_pil_list, boxes_list, processor, model):
    # boxes_list: list of tensors (N, 4)
    # SAM processor expects input_boxes as [[[x1, y1, x2, y2], ...], ...] (list of list of list)
    
    valid_indices = []
    input_boxes = []
    clean_images = []
    
    for i, boxes in enumerate(boxes_list):
        if len(boxes) > 0:
            valid_indices.append(i)
            input_boxes.append([boxes.tolist()]) # Extra nesting for "one list of boxes per image"
            clean_images.append(image_pil_list[i])
            
    if not clean_images:
        return [None] * len(image_pil_list)
        
    inputs = processor(clean_images, input_boxes=input_boxes, return_tensors="pt", padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    masks_list = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )
    
    # Re-assemble results
    final_results = [None] * len(image_pil_list)
    for i, idx in enumerate(valid_indices):
        final_results[idx] = masks_list[i]
        
    return final_results

def main():
    # 1. Setup
    if VIZ_DIR.exists(): shutil.rmtree(VIZ_DIR)
    VIZ_DIR.mkdir()
    
    output_ds_dir = Path("data") / OUTPUT_REPO_ID
    if output_ds_dir.exists(): shutil.rmtree(output_ds_dir)
    
    # 2. Load Dataset
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = LeRobotDataset(DATASET_NAME, root=".")
    
    # 3. Load Models
    gd_processor, gd_model, sam_processor, sam_model = load_models()
    
    # 4. Collect First Frames (Pre-fetch)
    logger.info("Collecting first frame of each episode for batch inference...")
    first_frames = []
    for ep_idx in range(ds.num_episodes):
        ep_meta = ds.meta.episodes[ep_idx]
        global_idx = ep_meta["dataset_from_index"]
        item = ds[global_idx]
        img_tensor = item[IMAGE_KEY]
        # Torch -> Numpy -> PIL
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy() if img_tensor.shape[0] == 3 else img_tensor.numpy()
        else:
            img_np = np.array(img_tensor)
            
        if img_np.max() <= 1.0: img_np = (img_np * 255).astype(np.uint8)
        else: img_np = img_np.astype(np.uint8)
        
        first_frames.append(Image.fromarray(img_np))
        
    logger.info(f"Collected {len(first_frames)} frames.")
    
    # 5. Batch Inference
    logger.info("Running Batch Inference (GroundingDINO + SAM)...")
    
    all_i_masks = []
    all_f_masks = []
    
    num_batches = (len(first_frames) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(first_frames))
        batch_imgs = first_frames[start:end]
        
        logger.info(f"Processing Batch {b+1}/{num_batches} (Ep {start}-{end-1})")
        
        # --- Tiger (Prompt I) ---
        res_i = get_bounding_boxes_batch(batch_imgs, PROMPT_I, gd_processor, gd_model)
        boxes_i = [r["boxes"] for r in res_i]
        masks_i_batch = segment_with_sam_batch(batch_imgs, boxes_i, sam_processor, sam_model)
        
        # --- Elephant (Prompt F) ---
        res_f = get_bounding_boxes_batch(batch_imgs, PROMPT_F, gd_processor, gd_model)
        boxes_f = [r["boxes"] for r in res_f]
        masks_f_batch = segment_with_sam_batch(batch_imgs, boxes_f, sam_processor, sam_model)
        
        # Post-process to single binary mask (H, W)
        for i in range(len(batch_imgs)):
            h, w = batch_imgs[i].size[::-1] # PIL size is W, H
            
            # Helper to combine SAM masks
            def combine(masks):
                if masks is None: return np.zeros((h, w), dtype=np.uint8)
                combined = np.zeros((h, w), dtype=bool)
                for m in masks:
                    # m is (1, H, W)? SAM output shape
                     combined = np.logical_or(combined, m[0].cpu().numpy())
                return (combined * 255).astype(np.uint8)
            
            all_i_masks.append(combine(masks_i_batch[i]))
            all_f_masks.append(combine(masks_f_batch[i]))
            
    # 6. Save Visualization (Batch Export)
    logger.info(f"Saving {len(all_i_masks)} mask pairs to {VIZ_DIR}...")
    for i in range(len(all_i_masks)):
        Image.fromarray(all_i_masks[i]).save(VIZ_DIR / f"ep_{i:02d}_mask_tiger.png")
        Image.fromarray(all_f_masks[i]).save(VIZ_DIR / f"ep_{i:02d}_mask_elephant.png")
        
    # 7. Write Dataset (IO Pass)
    logger.info("Writing new dataset...")
    
    # Prepare Features
    features = ds.features.copy()
    h, w = first_frames[0].size[::-1]
    features["observation.images.i_mask"] = {"dtype": "video", "shape": (1, h, w), "names": ["height", "width", "channels"]}
    features["observation.images.f_mask"] = {"dtype": "video", "shape": (1, h, w), "names": ["height", "width", "channels"]}
    
    new_ds_meta = LeRobotDataset.create(
        repo_id=OUTPUT_REPO_ID,
        fps=ds.fps,
        features=features,
        robot_type=ds.meta.robot_type,
        root=output_ds_dir,
        use_videos=True,
    )
    new_ds_meta.meta.info["chunks_size"] = 1000
    new_ds_meta.meta.metadata_buffer_size = 1000
    if ds.meta.tasks is not None and len(ds.meta.tasks) > 0: 
        new_ds_meta.meta.save_episode_tasks(ds.meta.tasks.index.tolist())
    
    for episode_idx in range(ds.num_episodes):
        if episode_idx % 5 == 0: logger.info(f"Writing Episode {episode_idx}/{ds.num_episodes}")
        
        ep_meta = ds.meta.episodes[episode_idx]
        from_idx = ep_meta["dataset_from_index"]
        to_idx = ep_meta["dataset_to_index"]
        num_frames = to_idx - from_idx
        
        # Prepare Masks (Repeated)
        # mask is (H, W), make it (T, 1, H, W)
        mask_i_arr = all_i_masks[episode_idx][None, None, ...].repeat(num_frames, axis=0)
        mask_f_arr = all_f_masks[episode_idx][None, None, ...].repeat(num_frames, axis=0)
        
        # Load other features
        episode_data = {k: [] for k in ds.features}
        for frame_idx in range(from_idx, to_idx):
            item = ds[frame_idx]
            for k in ds.features:
                val = item[k]
                if isinstance(val, torch.Tensor): val = val.numpy()
                episode_data[k].append(val)
        
        # Stack
        for k in episode_data: episode_data[k] = np.stack(episode_data[k])
        
        # Add Masks
        episode_data["observation.images.i_mask"] = mask_i_arr
        episode_data["observation.images.f_mask"] = mask_f_arr
        
        # Save Metadata
        new_ds_meta.meta.save_episode(
            episode_index=episode_idx,
            episode_length=num_frames,
            episode_tasks=ds.meta.episodes[episode_idx]["tasks"], 
            episode_stats={},
            episode_metadata={} 
        )
        
        # Save Videos (Manual injection logic)
        latest_ep_dict = new_ds_meta.meta.metadata_buffer[-1]
        chunk_idx = latest_ep_dict["meta/episodes/chunk_index"][0]
        file_idx = latest_ep_dict["meta/episodes/file_index"][0]
        
        video_keys = [k for k in features if features[k]["dtype"] in ["video", "image"]]
        
        for key in video_keys:
            latest_ep_dict[f"videos/{key}/chunk_index"] = [chunk_idx]
            latest_ep_dict[f"videos/{key}/file_index"] = [file_idx]
            latest_ep_dict[f"videos/{key}/from_timestamp"] = [0.0]
            latest_ep_dict[f"videos/{key}/to_timestamp"] = [(num_frames - 1) / ds.fps]
            
            vid_data = episode_data[key] # (T, C, H, W)
            vid_data_perm = np.transpose(vid_data, (0, 2, 3, 1)) # (T, H, W, C)
            
            # Format correction
            if vid_data_perm.dtype in [np.float32, np.float64]:
                 if vid_data_perm.max() <= 1.05: vid_data_perm = (vid_data_perm * 255).astype(np.uint8)
                 else: vid_data_perm = vid_data_perm.astype(np.uint8)
            elif vid_data_perm.dtype != np.uint8:
                 vid_data_perm = vid_data_perm.astype(np.uint8)
                 
            if vid_data_perm.shape[-1] == 1:
                vid_data_perm = np.repeat(vid_data_perm, 3, axis=-1)
                
            vid_rel_path = new_ds_meta.meta.video_path.format(video_key=key, chunk_index=chunk_idx, file_index=file_idx)
            vid_path = new_ds_meta.meta.root / vid_rel_path
            vid_path.parent.mkdir(parents=True, exist_ok=True)
            
            torchvision.io.write_video(str(vid_path), torch.from_numpy(vid_data_perm), fps=ds.fps, video_codec="libx264", options={"crf": "18"})
            
        # Write Parquet
        data_rel_path = new_ds_meta.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        data_path = new_ds_meta.meta.root / data_rel_path
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        parquet_data = {}
        for key in episode_data:
             if key not in video_keys:
                 parquet_data[key] = list(episode_data[key])
                 if isinstance(parquet_data[key][0], np.ndarray):
                     parquet_data[key] = [x.tolist() for x in parquet_data[key]]
                     
        start_frame = from_idx
        parquet_data["episode_index"] = [episode_idx] * num_frames
        parquet_data["frame_index"] = list(range(num_frames))
        parquet_data["timestamp"] = [i / ds.fps for i in range(num_frames)]
        parquet_data["index"] = list(range(start_frame, start_frame + num_frames))
        
        df = pd.DataFrame(parquet_data)
        if data_path.exists():
             existing_df = pd.read_parquet(data_path)
             df = pd.concat([existing_df, df], ignore_index=True)
        df.to_parquet(data_path, index=False)
        
    new_ds_meta._close_writer()
    logger.info("Done!")

if __name__ == "__main__":
    main()
