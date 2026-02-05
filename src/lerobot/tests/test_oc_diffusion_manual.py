
import torch
from lerobot.policies.oc_diffusion.configuration_oc_diffusion import OCDiffusionConfig
from lerobot.policies.oc_diffusion.modeling_oc_diffusion import OCDiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.configs.types import FeatureType, PolicyFeature

def test_oc_diffusion():
    # Setup Config
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
        # OC Params
        mask_feature_key="observation.masks",
        num_object_masks=2,
        vision_backbone="resnet18",
        crop_shape=(84, 84),
        # Reduce size for speed
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
    )
    
    # Initialize Policy
    print("Initializing Policy...")
    policy = OCDiffusionPolicy(config)
    
    # Create Dummy Batch
    B = 2
    T = config.n_obs_steps
    H_img, W_img = 96, 96
    
    batch = {
        "observation.state": torch.randn(B, T, 6),
        "observation.images": torch.randn(B, T, 3, H_img, W_img),
        "observation.masks": torch.rand(B, T, 2, H_img, W_img), # 2 objects
        "action": torch.randn(B, config.horizon, 6),
        "action_is_pad": torch.zeros(B, config.horizon, dtype=torch.bool)
    }
    
    # Test Forward (Loss)
    print("Testing Forward Pass...")
    loss, _ = policy(batch)
    print(f"Loss: {loss.item()}")
    
    # Test Action Selection
    print("Testing Action Selection...")
    policy.eval()
    
    # Reset queue
    policy.reset()
    
    # Need to populate queue first with n_obs_steps
    # select_action expects a single step observation usually, but the policy handles queueing.
    # We pass a batch of size B, but sequence length?
    # policy.select_action expects:
    # batch: {key: (B, ...)} NOT (B, T, ...) usually for inference loop if running env step by step.
    # BUT, the `modeling_diffusion.py` implementation of select_action does:
    # batch = {k: torch.stack(list(self._queues[k]), dim=1)...}
    # It expects the INPUT batch to be a single frame (B, ...) unless we provide history?
    # Actually `select_action` signature: `batch: dict[str, Tensor]`.
    # It calls `populate_queues(self._queues, batch)`.
    # `populate_queues` usually takes single step.
    
    # Let's mock a single step input
    single_step_batch = {
        "observation.state": torch.randn(B, 6),
        "observation.images": torch.randn(B, 3, H_img, W_img),
        "observation.masks": torch.rand(B, 2, H_img, W_img),
    }
    
    # Need to call reset to init queues
    policy.reset()
    
    # Feed n_obs_steps
    for _ in range(config.n_obs_steps):
         action = policy.select_action(single_step_batch)
    
    print(f"Action shape: {action.shape}")
    assert action.shape == (B, 6)
    
    print("Test Passed!")

if __name__ == "__main__":
    test_oc_diffusion()
