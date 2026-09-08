import duckdb
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import time

BNC_DIR = r"C:\Users\priha\OneDrive - Office's ID\Documents\BNC\2554\download\Texts"
DB_PATH = r"C:\Users\priha\Documents\cortex\corpora\english\BNC.duckdb"

def extract_header_metadata(root):
    meta = {
        'filename': '', 'title': '', 'publisher': '', 'pub_place': '',
        'pub_date': '', 'creation_date': '', 'class_code': '', 'keywords': '', 'cat_ref': ''
    }
    
    # Try to get file_id from bncDoc
    if 'id' in root.attrib:
        meta['filename'] = root.attrib['id']
    else:
        # Check xml:id
        for k, v in root.attrib.items():
            if k.endswith('id'):
                meta['filename'] = v

    header = root.find('.//teiHeader')
    if header is None:
        return meta

    # Title
    title_node = header.find('.//titleStmt/title')
    if title_node is not None and title_node.text:
        meta['title'] = title_node.text.strip()
    
    # idno type="bnc"
    idno_node = header.find('.//publicationStmt/idno[@type="bnc"]')
    if idno_node is not None and idno_node.text:
        meta['filename'] = idno_node.text.strip()
        
    # Imprint (Publisher, Place, Date)
    imprint = header.find('.//sourceDesc/bibl/imprint')
    if imprint is not None:
        pub_node = imprint.find('publisher')
        if pub_node is not None and pub_node.text: meta['publisher'] = pub_node.text.strip()
        
        place_node = imprint.find('pubPlace')
        if place_node is not None and place_node.text: meta['pub_place'] = place_node.text.strip()
        
        date_node = imprint.find('date')
        if date_node is not None:
            if 'value' in date_node.attrib: meta['pub_date'] = date_node.attrib['value']
            elif date_node.text: meta['pub_date'] = date_node.text.strip()

    # Creation Date
    creation_node = header.find('.//profileDesc/creation')
    if creation_node is not None:
        if 'date' in creation_node.attrib: meta['creation_date'] = creation_node.attrib['date']
        elif creation_node.text: meta['creation_date'] = creation_node.text.strip()

    # Text Class (ClassCode, Keywords, CatRef)
    text_class = header.find('.//profileDesc/textClass')
    if text_class is not None:
        class_node = text_class.find('classCode')
        if class_node is not None and class_node.text: meta['class_code'] = class_node.text.strip()
        
        kw_nodes = text_class.findall('.//keywords/term')
        if kw_nodes:
            meta['keywords'] = ", ".join([k.text.strip() for k in kw_nodes if k.text])
            
        cat_node = text_class.find('catRef')
        if cat_node is not None and 'targets' in cat_node.attrib:
            meta['cat_ref'] = cat_node.attrib['targets']

    return meta

def build_db():
    print(f"Creating DuckDB database at {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)
    
    xml_files = []
    for root_dir, dirs, files in os.walk(BNC_DIR):
        for f in files:
            if f.endswith('.xml'):
                xml_files.append(os.path.join(root_dir, f))

    total_files = len(xml_files)
    print(f"Found {total_files} XML files to process.")
    
    chunk_size = 1000000
    rows = []
    
    first_chunk = True
    start_time = time.time()
    
    for i, xml_file in enumerate(xml_files):
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i+1}/{total_files} files... ({elapsed:.1f}s)")
            
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            meta = extract_header_metadata(root)
            
            # Find all sentences
            for s in root.findall('.//s'):
                sentence_n = s.attrib.get('n', '')
                
                # Iterate all children (w and c)
                for child in s:
                    if child.tag in ('w', 'c'):
                        token = (child.text or "").strip()
                        if not token:
                            continue
                            
                        lemma = child.attrib.get('hw', '')
                        if not lemma and child.tag == 'c':
                            lemma = token
                            
                        pos_tag = child.attrib.get('c5', '')
                        wordclass = child.attrib.get('pos', '')
                        
                        row = {
                            'token': token,
                            '_token_low': token.lower(),
                            'lemma': lemma,
                            'pos': pos_tag,
                            'wordclass': wordclass,
                            'sentence_n': sentence_n
                        }
                        # Add metadata
                        row.update(meta)
                        rows.append(row)
                        
                        if len(rows) >= chunk_size:
                            df = pd.DataFrame(rows)
                            if first_chunk:
                                con.execute("CREATE TABLE corpus AS SELECT * FROM df")
                                first_chunk = False
                            else:
                                con.execute("INSERT INTO corpus SELECT * FROM df")
                            rows = []
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            
    # Insert remaining
    if rows:
        df = pd.DataFrame(rows)
        if first_chunk:
            con.execute("CREATE TABLE corpus AS SELECT * FROM df")
        else:
            con.execute("INSERT INTO corpus SELECT * FROM df")

    end_time = time.time()
    print(f"Database creation complete in {end_time - start_time:.1f} seconds.")
    
    count = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
    print(f"Total tokens in corpus: {count}")
    con.close()

if __name__ == "__main__":
    build_db()
