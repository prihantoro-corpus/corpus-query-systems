import xml.etree.ElementTree as ET
import duckdb
import pandas as pd
import argparse
import sys
import os
import re

def sanitize_xml(content_str):
    """Clean unescaped & symbols or XML formatting issues if any."""
    content_str = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', content_str)
    return content_str

def parse_uam_xml(xml_content, filename="corpus.xml"):
    """
    Parses UAM XML file offline, preserving inline <segment> markup & attributes.
    Returns (tokens_list, metadata_structure)
    """
    cleaned_xml = sanitize_xml(xml_content)
    root = ET.fromstring(cleaned_xml)
    
    lang_elem = root.find('.//header/lang')
    lang_code = lang_elem.text.strip() if lang_elem is not None and lang_elem.text else 'indonesian'
    
    body_elem = root.find('.//body')
    if body_elem is None:
        body_elem = root

    tokens_data = []
    tag_counters = {}
    sent_counter = 1

    def tokenize_text(text, context, sent_id):
        if not text or not text.strip():
            return
        
        # Split tokens cleanly preserving punctuation
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', text)
        raw_tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        
        for tok in raw_tokens:
            row = {
                'token': tok,
                'pos': 'TAG',
                'lemma': tok.lower(),
                'sent_id': sent_id,
                'filename': filename
            }
            row.update(context)
            tokens_data.append(row)

    def traverse(element, context, sent_id):
        nonlocal sent_counter
        current_context = context.copy()
        tag_name = element.tag.lower()
        
        structural_tags = {'corpus', 'text', 'document', 'body', 'header', 's', 'sent', 'p'}
        
        if tag_name not in structural_tags:
            current_context[f"in_{tag_name}"] = True
            tag_counters[tag_name] = tag_counters.get(tag_name, 0) + 1
            current_context[f"{tag_name}_id"] = str(element.attrib.get('id', tag_counters[tag_name]))
            
            attribs = dict(element.attrib)
            if tag_name == 'segment':
                if 'parent' not in attribs or not str(attribs['parent']).strip():
                    attribs['parent'] = "none"
                if 'domain' not in attribs:
                    attribs['domain'] = "none"
                if 'category' not in attribs:
                    attribs['category'] = "none"
                if 'subcategory' not in attribs:
                    attribs['subcategory'] = "none"
                if 'tag' not in attribs:
                    attribs['tag'] = "none"
                
                if 'features' in attribs:
                    feat_raw = attribs['features']
                    parts = feat_raw.split(';')
                    attribs['domain'] = parts[0].replace('-', '_') if len(parts) >= 1 and parts[0].strip() else "none"
                    attribs['category'] = parts[1].replace('-', '_') if len(parts) >= 2 and parts[1].strip() else "none"
                    attribs['subcategory'] = parts[2].replace('-', '_') if len(parts) >= 3 and parts[2].strip() else "none"
                    attribs['tag'] = parts[3].replace('-', '_') if len(parts) >= 4 and parts[3].strip() else "none"
                    attribs['features'] = feat_raw.replace(';', '_').replace('-', '_')

            for k, v in attribs.items():
                clean_v = str(v).replace(';', '_').strip()
                if not clean_v:
                    clean_v = "none"
                current_context[f"{tag_name}_{k.lower()}"] = clean_v

        tokens_before = len(tokens_data)
        
        if element.text:
            tokenize_text(element.text, current_context, sent_counter)
            
        for child in element:
            traverse(child, current_context, sent_counter)
            if child.tail:
                tokenize_text(child.tail, current_context, sent_counter)
                
        tokens_after = len(tokens_data)
        inner_len = tokens_after - tokens_before
        
        if tag_name not in structural_tags and inner_len > 0:
            tokens_data[tokens_before][f"in_{tag_name}_start"] = True
            tokens_data[tokens_before][f"{tag_name}_len"] = inner_len

    traverse(body_elem, {}, sent_counter)
    
    # Sentence ID assignment based on sentence ending punctuation
    curr_sent = 1
    for row in tokens_data:
        row['sent_id'] = curr_sent
        if row['token'] in ('.', '!', '?'):
            curr_sent += 1
            
    return tokens_data, lang_code

def convert_uam_to_duckdb(input_files, output_db_path, language="Indonesian", create_zip=False):
    """
    Converts one or more UAM XML files (or .zip archive containing XML files)
    into a CORTEX-compatible DuckDB database (.db) and optional .zip package.
    Runs 100% offline.
    """
    import zipfile
    import io

    all_tokens = []
    
    # 1. Gather all files (unpacking any .zip inputs)
    file_tuples = [] # (filename, content_string)
    
    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            continue
            
        if filepath.lower().endswith('.zip'):
            print(f"Unpacking ZIP archive: {filepath}...")
            with zipfile.ZipFile(filepath, 'r') as zf:
                for member in zf.infolist():
                    if not member.is_dir() and member.filename.lower().endswith(('.xml', '.txt', '.csv')) and not os.path.basename(member.filename).startswith('.'):
                        b_content = zf.read(member.filename)
                        content = b_content.decode('utf-8', errors='ignore')
                        file_tuples.append((os.path.basename(member.filename), content))
        else:
            fname = os.path.basename(filepath)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            file_tuples.append((fname, content))

    import json
    combined_structure = {}

    for fname, content in file_tuples:
        print(f"Processing: {fname}...")
        tokens, detected_lang = parse_uam_xml(content, filename=fname)
        all_tokens.extend(tokens)
        
        try:
            root = ET.fromstring(sanitize_xml(content))
            def process_elem(elem):
                tag = elem.tag.lower()
                if tag not in combined_structure:
                    combined_structure[tag] = {}
                for k, v in elem.attrib.items():
                    attr_k = k.lower()
                    if attr_k not in combined_structure[tag]:
                        combined_structure[tag][attr_k] = set()
                    if len(combined_structure[tag][attr_k]) < 50:
                        combined_structure[tag][attr_k].add(str(v))
                for child in elem:
                    process_elem(child)
            process_elem(root)
        except Exception as e:
            print(f"Warning: Could not extract XML structure for {fname}: {e}")

    if not all_tokens:
        print("Error: No tokens extracted.")
        sys.exit(1)

    df_corpus = pd.DataFrame(all_tokens)
    
    # Ensure standard mandatory CORTEX columns exist
    for col in ['token', 'pos', 'lemma', 'sent_id', 'filename']:
        if col not in df_corpus.columns:
            df_corpus[col] = "" if col in ('pos', 'lemma') else 0
            
    df_corpus['_token_low'] = df_corpus['token'].astype(str).str.lower()

    target_db_path = output_db_path
    if output_db_path.lower().endswith('.zip'):
        target_db_path = output_db_path[:-4] + ".db"
        create_zip = True

    if os.path.exists(target_db_path):
        try: os.remove(target_db_path)
        except: pass

    # Convert sets to lists for JSON serialization
    serializable_structure = {}
    for tag, attrs in combined_structure.items():
        serializable_structure[tag] = {k: list(v) for k, v in attrs.items()}

    # Ingest into DuckDB
    with duckdb.connect(target_db_path) as con:
        con.execute("CREATE TABLE corpus AS SELECT * FROM df_corpus")
        con.execute("ALTER TABLE corpus ADD COLUMN id INTEGER")
        con.execute("CREATE SEQUENCE seq_id START 1")
        con.execute("UPDATE corpus SET id = nextval('seq_id')")
        
        # Create standard CORTEX indexes
        con.execute("CREATE INDEX idx_token_low ON corpus(_token_low)")
        con.execute("CREATE INDEX idx_id ON corpus(id)")
        con.execute("CREATE INDEX idx_lemma ON corpus(lemma)")
        con.execute("CREATE INDEX idx_sent ON corpus(sent_id)")

        # Create CORTEX metadata table
        con.execute("CREATE TABLE corpus_metadata (key VARCHAR, value VARCHAR)")
        con.execute("INSERT INTO corpus_metadata VALUES ('language', ?)", [language])
        con.execute("INSERT INTO corpus_metadata VALUES ('tagger', 'Offline UAM XML Converter')")
        con.execute("INSERT INTO corpus_metadata VALUES ('xml_structure', ?)", [json.dumps(serializable_structure)])

    total_tokens = len(df_corpus)
    print(f"\nSuccess! Created CORTEX Database: '{target_db_path}' ({total_tokens:,} tokens)")

    # 2. Optionally package into a .zip archive
    if create_zip:
        zip_output_path = output_db_path if output_db_path.lower().endswith('.zip') else output_db_path + ".zip"
        print(f"Packaging database into ZIP archive: '{zip_output_path}'...")
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(target_db_path, arcname=os.path.basename(target_db_path))
        print(f"Package ZIP Created: '{zip_output_path}'")

def run_gui_mode():
    """Interactive File Explorer GUI mode when no command-line arguments are supplied."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Error: Tkinter is required for GUI file selection.")
        sys.exit(1)
        
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("=" * 60)
    print("  CORTEX Offline UAM XML / ZIP Corpus Converter")
    print("=" * 60)
    print("\nOpening File Explorer to select XML file(s) or ZIP archive...")
    
    selected_files = filedialog.askopenfilenames(
        title="Select UAM XML files or ZIP archive",
        filetypes=[
            ("Corpus Files (*.xml, *.zip)", "*.xml;*.zip"),
            ("XML Files (*.xml)", "*.xml"),
            ("ZIP Archives (*.zip)", "*.zip"),
            ("All Files (*.*)", "*.*")
        ]
    )
    
    if not selected_files:
        print("\nNo files selected. Operation cancelled.")
        return
        
    print(f"\nSelected {len(selected_files)} file(s):")
    for f in selected_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(selected_files) > 5:
        print(f"  ... and {len(selected_files) - 5} more files.")
        
    # Language input prompt
    try:
        lang_input = input("\nEnter Corpus Language [default: Indonesian]: ").strip()
    except EOFError:
        lang_input = "Indonesian"
        
    language = lang_input if lang_input else "Indonesian"
    
    # Select save location
    print("\nOpening File Explorer to choose where to save .zip package...")
    save_path = filedialog.asksaveasfilename(
        title="Save Corpus Package (.zip)",
        defaultextension=".zip",
        initialfile="cortex_uam_corpus_package.zip",
        filetypes=[
            ("ZIP Package (*.zip)", "*.zip"),
            ("DuckDB Database (*.db)", "*.db")
        ]
    )
    
    if not save_path:
        print("\nNo save location selected. Operation cancelled.")
        return
        
    print(f"\nProcessing and packaging corpus into: '{save_path}'...")
    convert_uam_to_duckdb(list(selected_files), save_path, language=language, create_zip=save_path.lower().endswith('.zip'))
    
    print("\n" + "=" * 60)
    print(f"  SUCCESS! Your corpus package is saved at:\n  {os.path.abspath(save_path)}")
    print("=" * 60 + "\n")
    
    try:
        input("Press ENTER to exit...")
    except EOFError:
        pass

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_gui_mode()
    else:
        parser = argparse.ArgumentParser(description="100% Offline Converter from UAM XML / ZIP Corpora to CORTEX DuckDB Database (.db / .zip)")
        parser.add_argument("input_files", nargs="+", help="Paths to UAM XML files or .zip archive")
        parser.add_argument("-o", "--output", default="uam_corpus.db", help="Output .db or .zip path (default: uam_corpus.db)")
        parser.add_argument("-l", "--lang", default="Indonesian", help="Language name (default: Indonesian)")
        parser.add_argument("-z", "--zip", action="store_true", help="Package output as a .zip file")
        
        args = parser.parse_args()
        convert_uam_to_duckdb(args.input_files, args.output, args.lang, create_zip=args.zip)
