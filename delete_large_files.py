import getpass
from huggingface_hub import HfApi

def main():
    print("=== Hugging Face Dataset Cleaner ===")
    print("This will delete any files larger than 1 GB from your dataset repository.")
    token = getpass.getpass("Paste your Hugging Face WRITE Token and press Enter: ")
    
    repo_id = "prihantoro-corpus/cortex-data"
    api = HfApi(token=token)
    
    print(f"\nScanning files in {repo_id}...")
    try:
        files = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
        large_files = []
        for file in files:
            # size is in bytes
            if getattr(file, 'size', 0) > 1024**3:
                large_files.append(file.path)
                
        if not large_files:
            print("✅ No files larger than 1 GB found on Hugging Face!")
            return
            
        print("\nFound the following large files (>1GB):")
        for lf in large_files:
            print(f" - {lf}")
            
        confirm = input("\nDo you want to permanently delete these from Hugging Face? (y/n): ")
        if confirm.lower() == 'y':
            for lf in large_files:
                print(f"Deleting {lf}...")
                api.delete_file(path_in_repo=lf, repo_id=repo_id, repo_type="dataset")
            print("✅ Deletion complete!")
        else:
            print("Canceled.")
    except Exception as e:
        print(f"❌ Error accessing repository: {e}")

if __name__ == "__main__":
    main()
