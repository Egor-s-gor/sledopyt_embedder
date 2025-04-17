import os
from huggingface_hub import HfApi, login, create_repo

# --- Configuration ---
local_model_folder = "/root/sledopyt/.../training_output/checkpoint-1920" # Replace with the actual path
hf_username = "George2002"  # Replace with your Hugging Face username
hf_repo_name = "sledopyt_embedder" # Choose a name for your repository on the Hub
hf_token = "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" # Optional: Replace with your write token, or use login() interactively

# --- 1. Login ---
# Option A: Interactive login (will prompt you)
# login()
# Option B: Login with a token (useful in scripts/non-interactive environments)
login(token=hf_token)

# --- 2. Create a Repository on the Hub (if it doesn't exist) ---
# The repo_id is typically "username/repo_name" or "organization/repo_name"
repo_id = f"{hf_username}/{hf_repo_name}"
create_repo(repo_id, exist_ok=True, repo_type="model") # exist_ok=True prevents error if repo already exists
                                                      # repo_type can be 'model', 'dataset', or 'space'
print(f"Repository created or already exists: https://huggingface.co/{repo_id}")

# --- 3. Upload the Folder ---
api = HfApi()
api.upload_folder(
    folder_path=local_model_folder,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload model checkpoint" # Optional commit message
    # Use allow_patterns="*.safetensors" or ignore_patterns="*.bin" to filter files if needed
)

print(f"Folder '{local_model_folder}' uploaded successfully to repository '{repo_id}'!")