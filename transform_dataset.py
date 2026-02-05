from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

import torch
from torchvision.transforms import v2
from PIL import Image, ImageDraw
import numpy as np


class FixedBoundingBoxTransform:
    """
    Custom transform that draws a fixed bounding box on each image.
    The bounding box coordinates are in (x1, y1, x2, y2) format.
    """
    def __init__(self, bbox=(50, 50, 200, 200), color="red", width=3):
        """
        Args:
            bbox: Tuple of (x1, y1, x2, y2) defining the bounding box
            color: Color of the bounding box (any PIL-supported color)
            width: Width of the bounding box lines
        """
        self.bbox = bbox
        self.color = color
        self.width = width
    
    def __call__(self, img):
        # Handle torch tensors
        if isinstance(img, torch.Tensor):
            # Convert to PIL Image
            # Assuming CHW format with values in [0, 1] or [0, 255]
            if img.max() <= 1.0:
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
            img = Image.fromarray(img_np)
        
        # Draw bounding box
        draw = ImageDraw.Draw(img)
        draw.rectangle(self.bbox, outline=self.color, width=self.width)
        
        # Convert back to tensor if needed
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        return img_tensor


# Create transform with fixed bounding box
# Adjust bbox coordinates (x1, y1, x2, y2) as needed for your images
torchvision_transforms_bb = v2.Compose([
    FixedBoundingBoxTransform(bbox=(50, 50, 200, 200), color="red", width=3),
])

dataset = LeRobotDataset(
    repo_id="RaspberryVitriol/bb",
    image_transforms=torchvision_transforms_bb
)
print("LOADED DATASET")
dataset.finalize()
print("FINALIZED DATASET")

dataset.push_to_hub("test")