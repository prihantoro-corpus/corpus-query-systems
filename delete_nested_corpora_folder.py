import gc
import os
import shutil

gc.collect()

nested_dir = r"c:\Users\priha\Documents\cortex\corpora\corpora"

if os.path.exists(nested_dir):
    # Try deleting individual files ignore_errors=True
    shutil.rmtree(nested_dir, ignore_errors=True)
    if not os.path.exists(nested_dir):
        print("Successfully deleted nested corpora folder!")
    else:
        print("Folder still present, attempting force remove on remaining files...")
        for root, dirs, files in os.walk(nested_dir, topdown=False):
            for f in files:
                fp = os.path.join(root, f)
                try: os.remove(fp)
                except: pass
            for d in dirs:
                dp = os.path.join(root, d)
                try: os.rmdir(dp)
                except: pass
        try: os.rmdir(nested_dir)
        except: pass
