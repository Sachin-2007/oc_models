
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Constants
REPO_ID = "RaspberryVitriol/oc_segment"
ROOT = Path("data")

print(f"Uploading {REPO_ID} from {ROOT}...")
try:
    ds = LeRobotDataset(REPO_ID, root=ROOT)
    ds.push_to_hub()
    print("Upload completed successfully!")
except Exception as e:
    print(f"Failed to upload: {e}")
