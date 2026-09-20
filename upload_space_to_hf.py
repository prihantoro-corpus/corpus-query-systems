import os
import sys
from huggingface_hub import HfApi

def upload_space():
    print("=" * 60)
    print(" HUGGING FACE SPACE DIRECT UPLOADER ")
    print("=" * 60)
    
    # Minta token jika belum ada
    token = input("Masukkan Hugging Face User Access Token (write token): ").strip()
    if not token:
        print("❌ Token tidak boleh kosong!")
        sys.exit(1)
        
    repo_id = "prihantoro-corpus/cortex"
    print(f"\n[INFO] Mengunggah seluruh folder proyek ke HF Space: {repo_id}...")
    
    # Pola file yang diabaikan (agar upload cepat & tidak melebihi limit)
    ignore_patterns = [
        ".git/*",
        ".github/*",
        "__pycache__/*",
        "*.pyc",
        "*.db",
        "*.duckdb",
        "*.wav",
        "*.whl",
        "*.docx",
        "*.pptx",
        "*.pdf",
        ".gemini/*",
        ".idea/*",
        ".vscode/*",
        "test_*.py",
        "scratch/*"
    ]
    
    api = HfApi(token=token)
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns
        )
        print("\n🚀 BERHASIL! Perubahan terbaru telah ter-push secara langsung ke Hugging Face Space!")
        print(f"Buka HF Space Anda di: https://huggingface.co/spaces/{repo_id}")
        print("Space akan otomatis melakukan rebuild dan merestart server Streamlit.")
    except Exception as e:
        print(f"\n❌ Upload gagal: {e}")

if __name__ == "__main__":
    upload_space()
