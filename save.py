from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi

# Step 2: Load source dataset
dataset = load_dataset("aadarshram/pick_place_tape")
dataset.save_to_disk("bb_local")

# Step 3: Ensure target repo exists
api = HfApi()
api.create_repo(repo_id="RaspberryVitriol/bb", repo_type="dataset", exist_ok=True)

# Step 4: Load from disk and push to your repo
dataset = load_from_disk("bb_local")
dataset.push_to_hub("RaspberryVitriol/bb")

# Step 5: Create version tag
api.create_tag("RaspberryVitriol/bb", tag="v1.0", repo_type="dataset")

# Step 6: Verify
print(api.list_repo_refs("RaspberryVitriol/bb", repo_type="dataset"))