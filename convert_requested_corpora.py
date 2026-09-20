import os
import sys

sys.path.append(r"c:\Users\priha\Documents\cortex")
from core.preprocessing.export_service import export_db_to_vertical_xml

corpora_dir = r"c:\Users\priha\Documents\cortex\corpora"

target_files = [
    os.path.join(corpora_dir, "arabic", "ar_pud-ud-test.db"),
    os.path.join(corpora_dir, "chinese", "zh_gsd-ud-train.db"),
    os.path.join(corpora_dir, "indonesian", "KOSLAT-full.db"),
    os.path.join(corpora_dir, "indonesian", "ID-BPPT-tagged.db"),
    os.path.join(corpora_dir, "indonesian", "tag_sample_elan_30_sentences.db"),
    os.path.join(corpora_dir, "english", "EN-BPPT-tagged.db"),
    os.path.join(corpora_dir, "english", "ICNALE_WE.db"),
    os.path.join(corpora_dir, "english", "sm_icnale.db"),
    os.path.join(corpora_dir, "korean", "ko_gsd-ud-train.db"),
    os.path.join(corpora_dir, "korean", "korean_sample.db")
]

print("Starting conversion of all requested databases to XML...")

for db_path in target_files:
    if os.path.exists(db_path):
        base, _ = os.path.splitext(db_path)
        xml_path = base + ".xml"
        rel_path = os.path.relpath(db_path, corpora_dir)
        xml_rel_path = os.path.relpath(xml_path, corpora_dir)
        
        print(f"Converting: {rel_path} -> {xml_rel_path} ...")
        try:
            xml_content = export_db_to_vertical_xml(db_path)
            if xml_content and not xml_content.startswith("Error"):
                os.makedirs(os.path.dirname(xml_path), exist_ok=True)
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(xml_content)
                print(f"  [OK] Successfully created {xml_rel_path} ({len(xml_content)} bytes)")
            else:
                print(f"  [FAIL] {xml_content[:100]}")
        except Exception as e:
            print(f"  [ERROR] {e}")
    else:
        print(f"[SKIP] DB file not found: {db_path}")

print("Targeted conversion completed!")
