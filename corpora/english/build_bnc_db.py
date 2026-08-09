import sqlite3
import os
import csv
import glob
import xml.etree.ElementTree as ET

BNC_DIR = r"C:\Users\priha\OneDrive - Office's ID\Documents\BNC\bnc2014spoken-xml"
SPEAKER_METADATA = os.path.join(BNC_DIR, "spoken", "metadata", "bnc2014spoken-speakerdata.tsv")
TEXT_METADATA = os.path.join(BNC_DIR, "spoken", "metadata", "bnc2014spoken-textdata.tsv")
UNTAGGED_DIR = os.path.join(BNC_DIR, "spoken", "untagged")
DB_PATH = r"C:\Users\priha\Documents\cortex\corpora\english\BNC Spoken.db"

def build_db():
    print(f"Creating database at {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE speakers (
            speaker_id TEXT PRIMARY KEY,
            exactage TEXT, age1994 TEXT, agerange TEXT, gender TEXT, nat TEXT, 
            birthplace TEXT, birthcountry TEXT, l1 TEXT, lingorig TEXT, 
            dialect_rep TEXT, hab_city TEXT, hab_country TEXT, hab_dur TEXT, 
            dialect_l1 TEXT, dialect_l2 TEXT, dialect_l3 TEXT, dialect_l4 TEXT, 
            edqual TEXT, occupation TEXT, socgrade TEXT, nssec TEXT, 
            l2 TEXT, fls TEXT, in_core TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE texts (
            text_id TEXT PRIMARY KEY,
            rec_length TEXT, rec_date TEXT, rec_year TEXT, rec_period TEXT, 
            n_speakers TEXT, list_speakers TEXT, rec_loc TEXT, relationships TEXT, 
            topics TEXT, activity TEXT, conv_type TEXT, conventions TEXT, 
            in_sample TEXT, transcriber TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE utterances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_id TEXT,
            speaker_id TEXT,
            n TEXT,
            utterance_text TEXT,
            FOREIGN KEY(text_id) REFERENCES texts(text_id),
            FOREIGN KEY(speaker_id) REFERENCES speakers(speaker_id)
        )
    ''')
    
    # Create FTS table
    cursor.execute('''
        CREATE VIRTUAL TABLE utterances_fts USING fts5(
            text_id UNINDEXED, 
            speaker_id UNINDEXED, 
            utterance_text, 
            content='utterances', 
            content_rowid='id'
        )
    ''')

    # Load speaker metadata
    print("Loading speaker metadata...")
    with open(SPEAKER_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = ['speaker_id', 'exactage', 'age1994', 'agerange', 'gender', 'nat', 
            'birthplace', 'birthcountry', 'l1', 'lingorig', 'dialect_rep', 'hab_city', 
            'hab_country', 'hab_dur', 'dialect_l1', 'dialect_l2', 'dialect_l3', 'dialect_l4', 
            'edqual', 'occupation', 'socgrade', 'nssec', 'l2', 'fls', 'in_core']
        placeholders = ', '.join(['?'] * len(headers))
        col_list = ', '.join(headers)
        
        insert_speaker_sql = f"INSERT INTO speakers ({col_list}) VALUES ({placeholders})"
        for row in reader:
            if len(row) == len(headers):
                cursor.execute(insert_speaker_sql, row)

    # Load text metadata
    print("Loading text metadata...")
    with open(TEXT_METADATA, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = ['text_id', 'rec_length', 'rec_date', 'rec_year', 'rec_period', 
            'n_speakers', 'list_speakers', 'rec_loc', 'relationships', 'topics', 
            'activity', 'conv_type', 'conventions', 'in_sample', 'transcriber']
        placeholders = ', '.join(['?'] * len(headers))
        col_list = ', '.join(headers)
        
        insert_text_sql = f"INSERT INTO texts ({col_list}) VALUES ({placeholders})"
        for row in reader:
            if len(row) == len(headers):
                cursor.execute(insert_text_sql, row)

    # Load utterances
    print("Loading utterances from XML files...")
    xml_files = glob.glob(os.path.join(UNTAGGED_DIR, '*.xml'))
    total_files = len(xml_files)
    
    utterance_count = 0
    for i, xml_file in enumerate(xml_files):
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{total_files} files...")
            
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text_id = root.attrib.get('id')
        
        # Get all <u> tags
        for u in root.findall('.//u'):
            speaker_id = u.attrib.get('who', '')
            n = u.attrib.get('n', '')
            # Get text recursively including text in inner tags (like <trunc> or <unclear>)
            # but omitting the tag names themselves.
            text_content = "".join(u.itertext()).strip()
            
            if text_content:
                cursor.execute('''
                    INSERT INTO utterances (text_id, speaker_id, n, utterance_text) 
                    VALUES (?, ?, ?, ?)
                ''', (text_id, speaker_id, n, text_content))
                utterance_count += 1

    print(f"Inserted {utterance_count} utterances.")

    # Populate FTS
    print("Populating FTS index...")
    cursor.execute('''
        INSERT INTO utterances_fts(rowid, text_id, speaker_id, utterance_text)
        SELECT id, text_id, speaker_id, utterance_text FROM utterances
    ''')

    # Create standard indexes
    print("Creating indexes...")
    cursor.execute("CREATE INDEX idx_utterances_text_id ON utterances(text_id)")
    cursor.execute("CREATE INDEX idx_utterances_speaker_id ON utterances(speaker_id)")

    conn.commit()
    conn.close()
    print("Database creation complete.")

if __name__ == "__main__":
    build_db()
