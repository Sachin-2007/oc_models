
import logging
import torch
from torch.utils.data import DataLoader
from lerobot.policies.oc_diffusion.configuration_oc_diffusion import OCDiffusionConfig
from lerobot.policies.oc_diffusion.modeling_oc_diffusion import OCDiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from termcolor import colored
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # Constants
    DATASET_REPO_ID = "aadarshram/pick_place_tiger_near_elephant_segmented"
    # Assuming dataset is local in 'data' directory relative to where we run or standard cache
    # If we created it in 'data/...' locally, LeRobotDataset(root='data') might find it?
    # Or strict path.
    # The segmentation script output to `data/aadarshram...`.
    # So root should be `data`.
    
    ROOT_DIR = Path("data")
    
    # 1. Config
    config = OCDiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        # Input features matching the dataset
        input_features={
            "observation.state": {"type": "STATE", "shape": (6,)}, # Verify shape!
            "observation.images.top_phone": {"type": "VISUAL", "shape": (3, 96, 96)},
        },
        output_features={
            "action": {"type": "ACTION", "shape": (6,)} # Verify shape!
        },
        mask_feature_keys=["observation.images.i_mask", "observation.images.f_mask"],
        num_object_masks=2,
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
        num_train_timesteps=50, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Training params
        optimizer_lr=1e-4,
    )
    
    # 2. Dataset
    # We need to verify shapes. LeRobotDataset handles loading.
    # We use the dataset created by segment script.
    logger.info("Loading dataset...")
    dataset = LeRobotDataset(DATASET_REPO_ID, root=ROOT_DIR)
    
    # Check features to update config if needed
    # (Optional auto-configuration or validation)
    
    # Create DataLoader
    # LeRobotDataset is a torch Dataset.
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4,
        drop_last=True
    )
    
    # 3. Policy
    policy = OCDiffusionPolicy(config)
    device = torch.device(config.device)
    policy.to(device)
    policy.train()
    
    # 4. Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.optimizer_lr)
    
    logger.info("Starting Training...")
    
    # 5. Training Loop
    epochs = 10 
    step_count = 0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward (computes loss)
            loss, _ = policy(batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            step_count += 1
            if i % 10 == 0:
                print(f"Step {step_count}: Loss = {loss.item():.4f}")
                
    print(colored("Training Finished Successfully!", "green"))
    
    # Save checkpoint?
    # policy.save_pretrained("outputs/train_oc_custom")

if __name__ == "__main__":
    train()
