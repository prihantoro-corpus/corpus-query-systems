import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.preprocessing.export_service import export_db_to_vertical_xml

corpora_dir = os.path.join(os.path.dirname(__file__), 'corpora')

print("Starting conversion of all .db/.duckdb files in corpora to .xml...")

converted_count = 0

for root, dirs, files in os.walk(corpora_dir):
    for f in files:
        if f.lower().endswith(('.db', '.duckdb')):
            db_path = os.path.join(root, f)
            base_name = os.path.splitext(f)[0]
            xml_path = os.path.join(root, f"{base_name}.xml")
            rel_path = os.path.relpath(db_path, corpora_dir)
            
            print(f"Processing: {rel_path}")
            
            try:
                xml_data = export_db_to_vertical_xml(db_path)
                if xml_data and not xml_data.startswith("Error"):
                    with open(xml_path, 'w', encoding='utf-8') as xml_out:
                        xml_out.write(xml_data)
                    print(f"  [SUCCESS] Converted: {rel_path} -> {base_name}.xml")
                    converted_count += 1
                else:
                    print(f"  [FAIL] Could not convert {rel_path}: {xml_data[:100]}")
            except Exception as e:
                print(f"  [FAIL] Exception converting {rel_path}: {e}")

print(f"\nConversion complete! Total converted: {converted_count}")
