# OC-Diffusion (Object-Centric Diffusion Policy)

## Overview

OC-Diffusion extends the standard Diffusion Policy by incorporating **object-centric features** from segmentation masks. The policy uses mask centroids as geometric tokens that provide explicit object position information to the action prediction network.

### Key Differences from Standard Diffusion Policy

| Component | Standard Diffusion | OC-Diffusion |
|-----------|-------------------|--------------|
| **Inputs** | state + images | state + images + **masks** |
| **Geometry** | N/A | `CentroidExtractor` extracts (x,y) from masks |
| **Embedding** | N/A | `SinusoidalPosEmb` encodes centroid positions |
| **Encoder** | N/A | `GeometryEncoder` (Conv + Pool) processes embeddings |

## Dataset Requirements

The dataset must include segmentation mask features alongside the standard image observations:

```python
features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    "observation.images.top_phone": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    # OC-Diffusion requires mask features:
    "observation.images.i_mask": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),  # Initial object
    "observation.images.f_mask": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),  # Final object
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
}
```

## Creating a Segmented Dataset

Use the `add_masks.py` utility to create a segmented dataset from an existing dataset:

```bash
conda activate lerobot

python -m lerobot.utils.add_masks \
    --repo-id aadarshram/pick_place_tiger_near_elephant \
    --output-repo-id RaspberryVitriol/oc_segment \
    --push-to-hub
```

This uses hybrid segmentation:
- **Tiger (i_mask)**: HSV color detection → SAM
- **Elephant (f_mask)**: GroundingDINO → SAM

## Training Command

### Quick Test (Synthetic Data)

```bash
conda activate lerobot

python -m lerobot.scripts.train_oc_diffusion_synthetic
```

### Training on Real Dataset

```bash
conda activate lerobot

python -m lerobot.scripts.train_oc_custom
```

Or using the standard training script with configuration:

```bash
conda activate lerobot

python -m lerobot.scripts.lerobot_train \
    --policy.type=oc_diffusion \
    --dataset.repo_id=RaspberryVitriol/oc_segment \
    --policy.mask_feature_keys='["observation.images.i_mask", "observation.images.f_mask"]' \
    --policy.num_object_masks=2 \
    --training.num_epochs=100 \
    --wandb.enable=true \
    --wandb.project=oc_diffusion
```

## Configuration Options

Key OC-Diffusion specific config parameters:

```python
OCDiffusionConfig(
    # OC-Diffusion specific
    mask_feature_keys=["observation.images.i_mask", "observation.images.f_mask"],
    num_object_masks=2,
    geometry_embed_dim=32,
    geometry_encoder_hidden_dim=64,
    
    # Standard diffusion params
    n_obs_steps=2,
    horizon=16,
    n_action_steps=8,
    vision_backbone="resnet18",
    ...
)
```

## Files

- **Policy**: `src/lerobot/policies/oc_diffusion/`
  - `configuration_oc_diffusion.py` - Config class
  - `modeling_oc_diffusion.py` - Model implementation
- **Training scripts**: `src/lerobot/scripts/`
  - `train_oc_diffusion_synthetic.py` - Synthetic data test
  - `train_oc_custom.py` - Real dataset training
- **Segmentation**: `src/lerobot/utils/`
  - `add_masks.py` - Dataset segmentation utility
  - `segment/` - Alternative segmentation approaches (archived)
