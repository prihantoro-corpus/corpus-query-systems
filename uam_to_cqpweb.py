import xml.etree.ElementTree as ET
import argparse
import sys
import os
import re
import zipfile
import io

def sanitize_xml(content_str):
    """Clean unescaped & symbols or XML formatting issues."""
    return re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', content_str)

def convert_uam_file_to_vrt(xml_content, filename="doc1.xml", include_pos=False):
    """
    Converts a single UAM XML string into CQPweb Vertical (VRT) XML format.
    Output: List of VRT lines (XML structural tags on own line, tokens as 1 token per line).
    """
    cleaned_xml = sanitize_xml(xml_content)
    root = ET.fromstring(cleaned_xml)
    
    doc_id = os.path.splitext(os.path.basename(filename))[0]
    lang_elem = root.find('.//header/lang')
    lang_code = lang_elem.text.strip() if lang_elem is not None and lang_elem.text else 'indonesian'
    
    body_elem = root.find('.//body')
    if body_elem is None:
        body_elem = root

    vrt_lines = []
    vrt_lines.append(f'<text id="{doc_id}" filename="{filename}" lang="{lang_code}">')

    sent_counter = 1
    in_sentence = False

    def start_sentence_if_needed():
        nonlocal in_sentence, sent_counter
        if not in_sentence:
            vrt_lines.append(f'<s id="{sent_counter}">')
            in_sentence = True

    def end_sentence_if_needed():
        nonlocal in_sentence, sent_counter
        if in_sentence:
            vrt_lines.append('</s>')
            in_sentence = False
            sent_counter += 1

    def emit_token(token_text):
        if not token_text or not token_text.strip():
            return
        
        # Tokenize preserving punctuation
        cleaned = re.sub(r'([^\w\s])', r' \1 ', token_text)
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        
        for tok in tokens:
            start_sentence_if_needed()
            if include_pos:
                pos = "TAG"
                lemma = tok.lower()
                vrt_lines.append(f"{tok}\t{pos}\t{lemma}")
            else:
                vrt_lines.append(tok)
            
            if tok in ('.', '!', '?'):
                end_sentence_if_needed()

    def traverse(element):
        tag_name = element.tag.lower()
        structural_tags = {'corpus', 'text', 'document', 'body', 'header', 's', 'sent', 'p'}
        
        if tag_name not in structural_tags:
            attribs = dict(element.attrib)
            
            # Ensure segment XML attributes default to 'none' if missing or empty
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
            
            # Sanitization for CQPweb classification category handles
            if 'features' in attribs:
                feat_raw = attribs['features']
                parts = feat_raw.split(';')
                
                attribs['domain'] = parts[0].replace('-', '_') if len(parts) >= 1 and parts[0].strip() else "none"
                attribs['category'] = parts[1].replace('-', '_') if len(parts) >= 2 and parts[1].strip() else "none"
                attribs['subcategory'] = parts[2].replace('-', '_') if len(parts) >= 3 and parts[2].strip() else "none"
                attribs['tag'] = parts[3].replace('-', '_') if len(parts) >= 4 and parts[3].strip() else "none"
                
                attribs['features'] = feat_raw.replace(';', '_').replace('-', '_')

            attr_str = ""
            for k, v in attribs.items():
                clean_v = str(v).replace(';', '_').strip()
                if not clean_v:
                    clean_v = "none"
                attr_str += f' {k.lower()}="{clean_v}"'
            vrt_lines.append(f'<{tag_name}{attr_str}>')

        if element.text:
            emit_token(element.text)
            
        for child in element:
            traverse(child)
            if child.tail:
                emit_token(child.tail)
                
        if tag_name not in structural_tags:
            vrt_lines.append(f'</{tag_name}>')

    traverse(body_elem)
    end_sentence_if_needed()
    vrt_lines.append('</text>')
    
    return "\n".join(vrt_lines)

def process_uam_to_cqpweb(input_files, output_path, include_pos=False):
    """
    Reads one or more UAM XML files (or .zip archive) and compiles into CQPweb VRT format.
    """
    file_tuples = [] # (filename, content_string)
    
    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            continue
            
        if filepath.lower().endswith('.zip'):
            print(f"Unpacking ZIP archive: {filepath}...")
            with zipfile.ZipFile(filepath, 'r') as zf:
                for member in zf.infolist():
                    if not member.is_dir() and member.filename.lower().endswith(('.xml', '.txt')) and not os.path.basename(member.filename).startswith('.'):
                        b_content = zf.read(member.filename)
                        content = b_content.decode('utf-8', errors='ignore')
                        file_tuples.append((os.path.basename(member.filename), content))
        else:
            fname = os.path.basename(filepath)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            file_tuples.append((fname, content))

    if not file_tuples:
        print("Error: No valid XML files found to process.")
        sys.exit(1)

    all_vrt_docs = []
    for fname, content in file_tuples:
        print(f"Converting to CQPweb VRT: {fname}...")
        vrt_doc = convert_uam_file_to_vrt(content, filename=fname, include_pos=include_pos)
        all_vrt_docs.append(vrt_doc)

    full_vrt_output = "\n\n".join(all_vrt_docs) + "\n"

    # Package output
    if output_path.lower().endswith('.zip'):
        print(f"\nPackaging CQPweb VRT into ZIP archive: '{output_path}'...")
        vrt_basename = os.path.splitext(os.path.basename(output_path))[0] + ".vrt"
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(vrt_basename, full_vrt_output.encode('utf-8'))
            
            p_flags = " -P pos -P lemma" if include_pos else ""
            readme_text = (
                "CQPweb Corpus Indexing Instructions\n"
                "====================================\n\n"
                "1. Upload or copy the .vrt file to your CQPweb / CWB server.\n"
                "2. When indexing in CQPweb Web UI or using cqp-encode, specify:\n"
                "   - Structural Tags:\n"
                "     text: id, filename, lang\n"
                "     s: id\n"
                "     segment: id, features, state, parent\n\n"
                "3. CLI Command Example (CWB cqp-encode):\n"
                "   cqp-encode -d /path/to/cwb/data -f " + vrt_basename + " -R /path/to/registry/corpus \\\n"
                "              -S text:0+id+filename+lang \\\n"
                "              -S s:0+id \\\n"
                "              -S segment:0+id+features+state+parent" + p_flags + "\n"
            )
            zf.writestr("CQPweb_README.txt", readme_text)
        print(f"Package ZIP Created: '{output_path}'")
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_vrt_output)
        print(f"\nSuccess! Created CQPweb VRT file: '{output_path}'")

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
    print("  UAM XML to CQPweb Vertical XML (VRT) Converter")
    print("=" * 60)
    print("\nOpening File Explorer to select UAM XML file(s) or ZIP archive...")
    
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
        
    print("\nOpening File Explorer to choose where to save CQPweb VRT / ZIP result...")
    save_path = filedialog.asksaveasfilename(
        title="Save CQPweb Vertical XML Result",
        defaultextension=".vrt",
        initialfile="uam_cqpweb_corpus.vrt",
        filetypes=[
            ("CQPweb Vertical XML (*.vrt)", "*.vrt"),
            ("ZIP Package (*.zip)", "*.zip"),
            ("XML File (*.xml)", "*.xml")
        ]
    )
    
    if not save_path:
        print("\nNo save location selected. Operation cancelled.")
        return
        
    print(f"\nProcessing and creating CQPweb VRT at: '{save_path}'...")
    process_uam_to_cqpweb(list(selected_files), save_path)
    
    print("\n" + "=" * 60)
    print(f"  SUCCESS! Your CQPweb file is saved at:\n  {os.path.abspath(save_path)}")
    print("=" * 60 + "\n")
    
    try:
        input("Press ENTER to exit...")
    except EOFError:
        pass

if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_gui_mode()
    else:
        parser = argparse.ArgumentParser(description="100% Offline Converter from UAM XML / ZIP Corpora to CQPweb Vertical XML (VRT) format")
        parser.add_argument("input_files", nargs="+", help="Paths to UAM XML files or .zip archive")
        parser.add_argument("-o", "--output", default="cqpweb_corpus.vrt", help="Output .vrt or .zip path (default: cqpweb_corpus.vrt)")
        
        args = parser.parse_args()
        process_uam_to_cqpweb(args.input_files, args.output)
