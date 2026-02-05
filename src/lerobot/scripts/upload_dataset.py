from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Repo ID to upload to")
    parser.add_argument("--root", type=str, default="data", help="Root data directory")
    args = parser.parse_args()

    # The issue is that LeRobotDataset constructor expects `root` to be the PARENT directory usually?
    # No, LeRobotDataset(repo_id, root=path) expects path to be the root containing the dataset OR the parent?
    # If I pass local_dir (e.g. data/RaspberryVitriol/oc_segment), LeRobotDataset might expect data/RaspberryVitriol/oc_segment/RaspberryVitriol/oc_segment ??
    # Let's verify.
    # If I pass root="data", and repo_id="RaspberryVitriol/oc_segment".
    # LeRobotDataset constructs self.root = Path(root) / repo_id (if I don't pass explicit path? wait)
    
    # In my training script I used:
    # ROOT_DIR = Path("data") / DATASET_REPO_ID
    # ds = LeRobotDataset(DATASET_REPO_ID, root=ROOT_DIR)
    
    # In `lerobot_dataset.py`:
    # self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    
    # So if I pass root="data/RaspberryVitriol/oc_segment", self.root is exactly that.
    # And inside that, it looks for `data/chunk-000...`.
    # Yes.
    
    dataset_path = Path(args.root) / args.repo_id
    if not dataset_path.exists():
         # Maybe user passed full path in root?
         if Path(args.root).exists() and (Path(args.root) / "meta").exists():
             dataset_path = Path(args.root)
         else:
             print(f"Error: Dataset not found at {dataset_path}")
             return

    logger.info(f"Loading local dataset from {dataset_path}...")
    ds = LeRobotDataset(args.repo_id, root=dataset_path)
    
    logger.info(f"Pushing to hub: {args.repo_id}...")
    ds.push_to_hub()
    logger.info("Done!")

if __name__ == "__main__":
    main()
