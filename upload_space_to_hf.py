import os
import getpass
from huggingface_hub import HfApi

def main():
    print("=== Hugging Face Space Direct Uploader ===")
    print("This will bypass GitHub completely and upload directly to your Space.")
    token = getpass.getpass("Right-click to paste your Hugging Face WRITE Token and press Enter: ").strip()
    
    if not token.startswith("hf_"):
        print("\n❌ Error: The token must start with 'hf_'. It seems you pasted something else or left it blank!")
        return
    
    repo_id = "prihantoro-corpus/cortex"
    api = HfApi(token=token)
    
    print(f"\nUploading local files to Space {repo_id}...")
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            # We ignore things we don't want uploaded to the space
            ignore_patterns=[
                ".git*", 
                ".venv*", 
                "__pycache__*", 
                "**/*.db", 
                "**/*.duckdb",
                "**/*.wav",
                "**/*.XML",
                "**/*.par",
                "**/*.dll",
                "**/*.lib",
                "**/*.docx",
                "**/*.pptx",
                "**/*.whl",
                "**/*.pdf",
                "**/*.xlsx",
                ".github*"
            ]
        )
        print("\n🎉 Upload complete! Your Space is now building.")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
