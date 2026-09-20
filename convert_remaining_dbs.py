import os
import sys

sys.path.append(r"c:\Users\priha\Documents\cortex")
from core.preprocessing.export_service import export_db_to_vertical_xml

corpora_dir = r"c:\Users\priha\Documents\cortex\corpora"

print("Converting remaining .db files to .xml locally...")

for root, dirs, files in os.walk(corpora_dir):
    for f in files:
        if f.lower().endswith(('.db', '.duckdb')):
            db_path = os.path.join(root, f)
            base_name = os.path.splitext(f)[0]
            xml_path = os.path.join(root, f"{base_name}.xml")
            
            # Skip if XML already exists
            if os.path.exists(xml_path):
                continue
                
            rel_path = os.path.relpath(db_path, corpora_dir)
            print(f"Converting: {rel_path} ...")
            
            try:
                xml_data = export_db_to_vertical_xml(db_path)
                if xml_data and not xml_data.startswith("Error"):
                    with open(xml_path, 'w', encoding='utf-8') as xml_out:
                        xml_out.write(xml_data)
                    print(f"  -> SUCCESS: {base_name}.xml created!")
                else:
                    print(f"  -> FAIL: {xml_data[:80]}")
            except Exception as e:
                print(f"  -> EXCEPTION: {e}")

print("All DB to XML conversions complete!")
