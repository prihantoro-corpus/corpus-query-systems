import sys
import os
import shutil

# Add project root to path
# CWD is c:\Users\priha\Documents\cortex
sys.path.insert(0, os.getcwd())

from core.config import get_available_corpora, CORPORA_DIR, KNOWN_CORPORA_MAP

print(f"CORPORA_DIR: {CORPORA_DIR}")
print(f"Directory exists: {os.path.exists(CORPORA_DIR)}")
if os.path.exists(CORPORA_DIR):
    print(f"Contents: {os.listdir(CORPORA_DIR)}")

available = get_available_corpora()
print(f"Available Corpora: {available}")
print(f"Known Corpora Map: {KNOWN_CORPORA_MAP}")
