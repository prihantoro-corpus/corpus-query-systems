import duckdb
import os
import csv
import glob
import xml.etree.ElementTree as ET
import pandas as pd

BNC_DIR = r"C:\Users\priha\OneDrive - Office's ID\Documents\BNC\bnc2014spoken-xml"
SPEAKER_METADATA = os.path.join(BNC_DIR, "spoken", "metadata", "bnc2014spoken-speakerdata.tsv")
TEXT_METADATA = os.path.join(BNC_DIR, "spoken", "metadata", "bnc2014spoken-textdata.tsv")
TAGGED_DIR = os.path.join(BNC_DIR, "spoken", "tagged")
DB_PATH = r"C:\Users\priha\Documents\cortex\corpora\english\BNC Spoken.duckdb"

def build_db():
    print(f"Creating DuckDB database at {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = duckdb.connect(DB_PATH)

    # Load speaker metadata into dict
    speakers = {}
    speaker_cols = ['exactage', 'age1994', 'agerange', 'gender', 'nat', 
                    'birthplace', 'birthcountry', 'l1', 'lingorig', 'dialect_rep', 'hab_city', 
                    'hab_country', 'hab_dur', 'dialect_l1', 'dialect_l2', 'dialect_l3', 'dialect_l4', 
                    'edqual', 'occupation', 'socgrade', 'nssec', 'l2', 'fls', 'in_core']
    with open(SPEAKER_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 0:
                speaker_id = row[0]
                speakers[speaker_id] = {speaker_cols[i]: row[i+1] for i in range(len(speaker_cols)) if i+1 < len(row)}

    # Load text metadata into dict
    texts = {}
    text_cols = ['rec_length', 'rec_date', 'rec_year', 'rec_period', 
                 'n_speakers', 'list_speakers', 'rec_loc', 'relationships', 'topics', 
                 'activity', 'conv_type', 'conventions', 'in_sample', 'transcriber']
    with open(TEXT_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 0:
                text_id = row[0]
                texts[text_id] = {text_cols[i]: row[i+1] for i in range(len(text_cols)) if i+1 < len(row)}

    xml_files = glob.glob(os.path.join(TAGGED_DIR, '*.xml'))
    total_files = len(xml_files)
    
    # We will accumulate rows and write them in chunks using pandas
    chunk_size = 500000
    rows = []
    
    print("Parsing XML and inserting into DuckDB...")
    
    first_chunk = True
    for i, xml_file in enumerate(xml_files):
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{total_files} files...")
            
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text_id = root.attrib.get('id', '')
        text_meta = texts.get(text_id, {k: "" for k in text_cols})
        
        for u in root.findall('.//u'):
            speaker_id = u.attrib.get('who', '')
            speaker_meta = speakers.get(speaker_id, {k: "" for k in speaker_cols})
            
            for w in u.findall('.//w'):
                token = "".join(w.itertext()).strip()
                pos = w.attrib.get('pos', '')
                lemma = w.attrib.get('lemma', '')
                w_class = w.attrib.get('class', '')
                usas = w.attrib.get('usas', '')
                
                if token:
                    row = {
                        'token': token,
                        '_token_low': token.lower(),
                        'lemma': lemma,
                        'pos': pos,
                        'wordclass': w_class,
                        'usas': usas,
                        'filename': text_id,
                        'speaker_id': speaker_id
                    }
                    row.update(text_meta)
                    row.update(speaker_meta)
                    rows.append(row)
                    
                    if len(rows) >= chunk_size:
                        df = pd.DataFrame(rows)
                        if first_chunk:
                            con.execute("CREATE TABLE corpus AS SELECT * FROM df")
                            first_chunk = False
                        else:
                            con.execute("INSERT INTO corpus SELECT * FROM df")
                        rows = []
    
    # Insert remaining
    if rows:
        df = pd.DataFrame(rows)
        if first_chunk:
            con.execute("CREATE TABLE corpus AS SELECT * FROM df")
        else:
            con.execute("INSERT INTO corpus SELECT * FROM df")

    print("Database creation complete.")
    
    count = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
    print(f"Total tokens in corpus: {count}")
    con.close()

if __name__ == "__main__":
    build_db()
