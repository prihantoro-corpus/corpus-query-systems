from huggingface_hub import HfApi
import os
import sys

def upload_dataset():
    print("=" * 60)
    print(" HUGGING FACE DATASET DIRECT UPLOADER (DESTINATION: corpora/) ")
    print("=" * 60)
    
    token = input("Masukkan Hugging Face User Access Token (write token): ").strip()
    if not token:
        print("❌ Token tidak boleh kosong!")
        sys.exit(1)
        
    dataset_repo = "prihantoro-corpus/cortex-data"
    print(f"\n[INFO] Mengunggah folder 'corpora' lokal tepat ke subfolder 'corpora/' di HF Dataset '{dataset_repo}'...")
    
    # Abaikan file biner non-xml
    ignore_patterns = [
        "*.db",
        "*.duckdb",
        "*.whl",
        "*.docx",
        "*.pptx",
        "*.pdf",
        "*.TextGrid",
        "*.wav",
        "*.mp3",
        "*.zip",
        "*.tar",
        "*.gz",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "spoken/*"
    ]
    
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path="corpora",
            path_in_repo="corpora",
            repo_id=dataset_repo,
            repo_type="dataset",
            token=token,
            ignore_patterns=ignore_patterns
        )
        print("\n🚀 BERHASIL! Seluruh file korpus XML berhasil diunggah ke subfolder 'corpora/' di Hugging Face Dataset!")
        print(f"Lihat Dataset di: https://huggingface.co/datasets/{dataset_repo}/tree/main/corpora")
    except Exception as e:
        print(f"\n❌ Dataset upload error: {e}")

if __name__ == "__main__":
    upload_dataset()
