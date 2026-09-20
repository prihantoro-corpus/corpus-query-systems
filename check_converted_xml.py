import os

corpora_dir = r"c:\Users\priha\Documents\cortex\corpora"

xml_files = []
db_files = []

for root, dirs, files in os.walk(corpora_dir):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), corpora_dir)
        if f.lower().endswith('.xml'):
            xml_files.append(rel)
        elif f.lower().endswith(('.db', '.duckdb')):
            db_files.append(rel)

print(f"Total .xml files in corpora: {len(xml_files)}")
for x in xml_files:
    print(f"  - {x}")

print(f"\nTotal .db/.duckdb files in corpora: {len(db_files)}")
for d in db_files:
    print(f"  - {d}")
