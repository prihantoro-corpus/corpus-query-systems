import os
import sys
import platform
import subprocess
import tempfile
import threading
import requests

def get_ollama_download_url():
    """
    Detects the current operating system and returns the download URL for Ollama.
    Returns: (str, str) -> (URL, File Extension) or (None, None) if unsupported.
    """
    sys_name = platform.system()
    if sys_name == "Windows":
        return "https://ollama.com/download/OllamaSetup.exe", ".exe"
    elif sys_name == "Darwin":
        return "https://ollama.com/download/Ollama-darwin.zip", ".zip"
    else:
        return None, None

def is_ollama_installed():
    """
    Checks if Ollama is installed on the system.
    Returns: bool
    """
    import shutil
    
    # 1. Check if 'ollama' executable is in the system PATH
    if shutil.which('ollama') is not None:
        return True

    # 2. Check standard platform-specific directories
    sys_name = platform.system()
    if sys_name == "Windows":
        local_app_data = os.environ.get('LOCALAPPDATA', '')
        if local_app_data:
            win_path = os.path.join(local_app_data, 'Programs', 'Ollama', 'ollama.exe')
            if os.path.exists(win_path):
                return True
        program_files = os.environ.get('ProgramFiles', '')
        if program_files:
            win_path2 = os.path.join(program_files, 'Ollama', 'ollama.exe')
            if os.path.exists(win_path2):
                return True
    elif sys_name == "Darwin":
        mac_path = '/Applications/Ollama.app'
        if os.path.exists(mac_path):
            return True
            
    return False


class OllamaDownloadThread(threading.Thread):
    def __init__(self, url, dest_path):
        super().__init__()
        self.url = url
        self.dest_path = dest_path
        self.progress = 0.0
        self.status = "Starting download..."
        self.cancelled = False
        self.completed = False
        self.error = None
        # Make it a daemon thread so it exits if Streamlit process exits
        self.daemon = True

    def run(self):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(self.url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()

            total_length = response.headers.get('content-length')
            if total_length is None:
                with open(self.dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if self.cancelled:
                            break
                        if chunk:
                            f.write(chunk)
                if self.cancelled:
                    self._cleanup()
                else:
                    self.progress = 1.0
                    self.status = "Download completed."
                    self.completed = True
            else:
                total_length = int(total_length)
                dl = 0
                with open(self.dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if self.cancelled:
                            break
                        if chunk:
                            f.write(chunk)
                            dl += len(chunk)
                            self.progress = min(dl / total_length, 0.99)
                            self.status = f"Downloading: {dl / (1024*1024):.1f} MB / {total_length / (1024*1024):.1f} MB"
                
                if self.cancelled:
                    self._cleanup()
                else:
                    self.progress = 1.0
                    self.status = "Download completed."
                    self.completed = True

        except Exception as e:
            self.error = str(e)
            self._cleanup()

    def _cleanup(self):
        try:
            if os.path.exists(self.dest_path):
                os.remove(self.dest_path)
        except Exception:
            pass

def run_ollama_installer(installer_path):
    """
    Launches the installer in a detached process.
    Returns: (bool, str) -> (Success, Error Message)
    """
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            if hasattr(os, 'startfile'):
                os.startfile(installer_path)
            else:
                subprocess.Popen([installer_path], shell=True)
            return True, None
        elif sys_name == "Darwin":
            parent_dir = os.path.dirname(installer_path)
            subprocess.Popen(["open", parent_dir])
            return True, "macOS ZIP downloaded. Please unzip the installer and move Ollama to Applications."
        else:
            return False, "Unsupported platform for auto-execution. Please run the installer manually."
    except Exception as e:
        return False, f"Failed to run installer: {e}"
