import os

corpora_dir = r"c:\Users\priha\Documents\cortex\corpora"

print("Checking and fixing BAWE file naming and duplicates...")

for root, dirs, files in os.walk(corpora_dir):
    for f in files:
        if f.lower() == 'bawe.xml':
            old_path = os.path.join(root, f)
            new_path = os.path.join(root, "BAWE.xml")
            
            # Temporary rename if Windows case collision
            if f != "BAWE.xml":
                tmp_path = os.path.join(root, "BAWE_temp.xml")
                os.rename(old_path, tmp_path)
                os.rename(tmp_path, new_path)
                print(f"  Renamed: {f} -> BAWE.xml in {root}")

# Check for duplicate BAWE outside corpora/english
for root, dirs, files in os.walk(corpora_dir):
    rel = os.path.relpath(root, corpora_dir).replace('\\', '/')
    for f in files:
        if f.lower() in ('bawe.xml', 'bawe.db', 'bawe.duckdb') and rel != 'english':
            dup_path = os.path.join(root, f)
            try:
                os.remove(dup_path)
                print(f"  Removed duplicate file: {dup_path}")
            except Exception as e:
                print(f"  Error removing duplicate {dup_path}: {e}")

print("BAWE file cleanup complete!")
