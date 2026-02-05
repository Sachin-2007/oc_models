
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from termcolor import colored

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.oc_diffusion.configuration_oc_diffusion import OCDiffusionConfig
from lerobot.policies.oc_diffusion.modeling_oc_diffusion import OCDiffusionPolicy
from lerobot.optim.factory import make_optimizer_and_scheduler

# Setup simple logger
logging.basicConfig(level=logging.INFO)

class SyntheticOCDataset(Dataset):
    def __init__(self, length=100, n_obs_steps=2, horizon=16, h=96, w=96):
        self.length = length
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.h = h
        self.w = w

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Create dummy data matching expected shapes
        # Config expects:
        # state: (6,)
        # images: (3, 96, 96)
        # masks: (2, 96, 96)
        # action: (6,)
        
        # But wait, the policy usually handles the stacking of history if we rely on queues?
        # NO, during training (offline), the dataset must provide the history window.
        # "observation.state": (n_obs_steps, state_dim)
        
        return {
            "observation.state": torch.randn(self.n_obs_steps, 6),
            "observation.images": torch.randn(self.n_obs_steps, 3, self.h, self.w),
            "observation.masks": torch.rand(self.n_obs_steps, 2, self.h, self.w),
            "action": torch.randn(self.horizon, 6),
            "action_is_pad": torch.zeros(self.horizon, dtype=torch.bool),
            "index": idx
        }

def train_synthetic():
    # 1. Config
    config = OCDiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))
        },
        mask_feature_key="observation.masks",
        num_object_masks=2,
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        down_dims=(32, 64), # Small for speed
        diffusion_step_embed_dim=32,
        num_train_timesteps=10, # small for speed
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. Policy
    policy = OCDiffusionPolicy(config)
    
    # 3. Dataset & Dataloader
    dataset = SyntheticOCDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 4. Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    # 5. Training Loop
    device = torch.device(config.device)
    policy.to(device)
    policy.train()
    
    print(colored("Starting Synthetic Training...", "green"))
    
    for epoch in range(1):
        for i, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward
            loss, _ = policy(batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                print(f"Step {i}: Loss = {loss.item():.4f}")
                
            if i >= 10: # limit steps
                break
                
    print(colored("Training Finished Successfully!", "green"))

if __name__ == "__main__":
    train_synthetic()
