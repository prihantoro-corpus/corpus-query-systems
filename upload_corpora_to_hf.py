import os
import getpass
from huggingface_hub import HfApi, create_repo

def main():
    print("=== Hugging Face Corpora Uploader ===")
    print("This script will upload your 11GB of database files to a Hugging Face Dataset.")
    
    token = getpass.getpass("Right-click to paste your Hugging Face WRITE Token and press Enter: ")
    
    repo_id = "prihantoro-corpus/cortex-data"
    
    api = HfApi(token=token)
    
    print(f"\nCreating dataset repository {repo_id}...")
    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True, private=False)
        print("✅ Repository created or already exists.")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    corpora_path = "corpora"
    if not os.path.exists(corpora_path):
        print(f"❌ Error: Could not find folder '{corpora_path}' in the current directory.")
        return

    print(f"\nScanning '{corpora_path}' to exclude files larger than 1 GB...")
    ignore_patterns = []
    for root, dirs, files in os.walk(corpora_path):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.getsize(file_path) > 1024**3:
                # Get path relative to the corpora folder for Hugging Face ignore pattern
                rel_path = os.path.relpath(file_path, corpora_path).replace("\\", "/")
                ignore_patterns.append(rel_path)
                print(f" - Excluding oversized file: {rel_path}")

    print(f"\nUploading files from local '{corpora_path}' folder to {repo_id}...")
    print("This will take a while. Do not close this window!")
    
    try:
        api.upload_folder(
            folder_path=corpora_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="corpora",
            allow_patterns=["*.db", "*.duckdb", "*.xml", "*.txt"],
            ignore_patterns=ignore_patterns if ignore_patterns else None
        )
        print("\n🎉 Upload complete!")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
