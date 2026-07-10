import os

# Local corpora directory
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA_DIR = os.path.join(_ROOT_DIR, 'corpora')
TAGSET_DIR = os.path.join(_ROOT_DIR, 'tagset')

# Metadata mapping for known corpora (Display Name -> Relative Path from CORPORA_DIR)
KNOWN_CORPORA_MAP = {
    "XML Tag Demo (EN)": "english/xml_tag_demo.db",
    "ID-BPPT (XML Tagged)": "indonesian/ID-BPPT-tagged.db",
    "EN-BPPT (XML Tagged)": "english/EN-BPPT-tagged.db",
    "Brown 50% Only (XML EN TAGGED)": "english/BrownCorpus.db",
    "KOSLAT-ID (XML Tagged)": "indonesian/KOSLAT-full.db",
    "BAWE sample (English)": "english/BAWE.db",
}

# Map of built-in files that should be downloaded if missing (e.g. from GitHub Releases)
DOWNLOADABLE_ASSETS_MAP = {
    "english/BAWE.db": "https://github.com/prihantoro-corpus/cortex/releases/download/v.1.1.1-data/BAWE.db"
}

# Alias for backward compatibility
BUILT_IN_CORPORA = KNOWN_CORPORA_MAP

def get_available_corpora():
    """
    Returns a dictionary of Available Corpus Name -> Filename.
    Scans CORPORA_DIR and maps to known display names where possible.
    """
    available = {}
    
    # Log for debugging
    log_file = os.path.join(_ROOT_DIR, "corpora_scan.log")
    with open(log_file, "w") as f:
        f.write(f"Scanning CORPORA_DIR: {CORPORA_DIR}\n")
        
        if not os.path.exists(CORPORA_DIR):
            f.write("ERROR: CORPORA_DIR does not exist.\n")
            return {}
            
        # Reverse map for easy lookup: filename -> nice name
        filename_to_name = {v: k for k, v in KNOWN_CORPORA_MAP.items()}
        
        # Recursive walk
        count = 0
        for root, dirs, files in os.walk(CORPORA_DIR):
            for file_name in files:
                if file_name.lower().endswith(('.xml', '.txt', '.csv', '.xlsx', '.db', '.duckdb')):
                    full_path = os.path.join(root, file_name)
                    
                    # Get relative path from CORPORA_DIR
                    rel_path = os.path.relpath(full_path, CORPORA_DIR)
                    # Normalize path separators for comparison
                    rel_path_normalized = rel_path.replace('\\', '/')
                    
                    f.write(f"Found: {rel_path_normalized}\n")
                    
                    # Check if this relative path matches a known corpus
                    if rel_path_normalized in filename_to_name:
                        display_name = filename_to_name[rel_path_normalized]
                        available[display_name] = rel_path
                    else:
                        # Use relative path as display name for unknown files
                        if root == CORPORA_DIR:
                             available[file_name] = file_name
                        else:
                             # e.g. "indonesian/sample.txt"
                             available[rel_path_normalized] = rel_path
                    count += 1
        f.write(f"Total corpora found: {count}\n")

        # Also add downloadable corpora that are not present locally
        for display_name, rel_path in KNOWN_CORPORA_MAP.items():
            if display_name not in available:
                if rel_path in DOWNLOADABLE_ASSETS_MAP:
                    available[display_name] = rel_path

    return available



BUILT_IN_CORPUS_DETAILS = {
    "XML Tag Demo (EN)":
        """
        A **demo corpus** showcasing XML tag-based search capabilities. Contains 12 sentences with rich inline markup including person/place names (`<PN>`), organizations (`<ORG>`), numbers (`<NUM>`), evaluative language (`<EVAL>`), and technical terms.
        <br><br>
        **Use this to test**: `<PN type="person">`, `<EVAL sentiment="positive">`, `at <ORG type="university">`, etc.
        <br><br>
        **Guide**: See `XML_TAG_DEMO_GUIDE.md` in the english folder.
        """,
    "ID-BPPT (XML Tagged)": 
        """
        The **ID-BPPT Corpus** is a tagged Indonesian corpus (POS/Lemma). 
        <br><br>
        **Source:** BPPT (Badan Pengkajian dan Penerapan Teknologi).
        """,
    "EN-BPPT (XML Tagged)":
        """
        The **EN-BPPT Corpus** is a tagged English corpus (POS/Lemma) used for parallel experiments or monolingual analysis.
        <br><br>
        **Source:** BPPT.
        """,

    "Brown 50% Only (XML EN TAGGED)":
        """
        A 50% subsample of the Brown Corpus, the first million-word electronic corpus of English. This sample is provided in a **TreeTagger-style XML format** containing token, POS, and lemma.
        <br><br>
        **Source/Citation:** Francis, W. N., & Kučera, H. (1979). **Brown Corpus Manual: Standard Corpus of Present-Day Edited American English for Use with Digital Computers.** Brown University.
        """,
    "KOSLAT-ID (XML Tagged)":
        """
        KOSLAT-ID v.1.0 is the first narrative-annotated corpus of reviews of healthcare facilities in Indonesia. It is provided in a **tagged XML format** (token, POS, lemma).
        <br><br>
        **Source/Citation:** Prihantoro., Yuliawati, S., Ekawati, D., & Rachmat, A. (2026-in press). **KOSLAT-ID v.1.0: The first narrative-annotated corpus of reviews of healthcare facilities in Indonesia.** [Corpora, 21(1), xx–xx.](https://www.prihantoro.com)
        """,
    "BAWE sample (English)":
        """
        The **British Academic Written English (BAWE)** corpus contains proficient undergraduate and master’s level writing in various disciplines. This sample includes academic essays and reports.
        <br><br>
        **Source:** Nesi, H., Gardner, S., Thompson, P. & Wickens, P. (2008). **British Academic Written English corpus.** Coventry University.
        """,

}

STANZA_LANG_MAP = {
    "English": "en",
    "Indonesian": "id",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Greek": "el",
    "Russian": "ru",
    "Polish": "pl",
    "Ukrainian": "uk",
    "Romanian": "ro",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "nb",
    "Finnish": "fi",
    "Catalan": "ca",
    "Croatian": "hr",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Slovenian": "sl"
}
