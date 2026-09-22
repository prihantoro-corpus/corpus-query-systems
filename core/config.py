import os

# Local corpora directory
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPORA_DIR = os.path.join(_ROOT_DIR, 'corpora')
TAGSET_DIR = os.path.join(_ROOT_DIR, 'tagset')

# Metadata mapping for known corpora (Display Name -> Relative Path from CORPORA_DIR)
KNOWN_CORPORA_MAP = {
    "XML Tag Demo (EN)": "english/xml_tag_demo.xml",
    "ID-BPPT (XML Tagged)": "indonesian/ID-BPPT-tagged.xml",
    "EN-BPPT (XML Tagged)": "english/EN-BPPT-tagged.xml",
    "BROWN": "english/BrownCorpus.xml",
    "KOSLAT-ID (XML Tagged)": "indonesian/KOSLAT-full.xml",
    "BAWE sample (English)": "english/BAWE.duckdb",
    "JP-DICO-JALF": "japanese/JP-DICO-JALF.xml",
    "ICNALE Written English": "english/ICNALE_WE.xml",
    "ICNALE Spoken Monologue": "english/sm_icnale.xml",
    "BNC Spoken": "english/BNC Spoken.xml",
    "Arabic PUD UD Sample": "arabic/ar_pud-ud-test.xml",
    "Chinese GSD UD Sample": "chinese/zh_gsd-ud-train.xml",
    "Japanese GSD UD Sample": "japanese/ja_gsd-ud-train.xml",
    "Korean GSD UD Sample": "korean/ko_gsd-ud-train.xml",
    "Korean Sample (XML)": "korean/korean_sample.xml",
    "ELAN Indonesian Sample (XML)": "indonesian/tag_sample_elan_30_sentences.xml",
    "sample of ELAN file (Indonesian)": "indonesian/sample_elan_30_sentences.eaf",
    "sample of ELAN file (Korean)": "korean/sample_korean_30_sentences.eaf",
    "ITA3 Spoken Sample": "spoken/ITA3.TextGrid",
    "TUFS2023KOMSHI314 (UAM XML)": "indonesian/uam_trial/TUFS2023KOMSHI314.xml",
}

# Map of built-in files that should be downloaded if missing (e.g. from GitHub Releases)
DOWNLOADABLE_ASSETS_MAP = {}

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
            
        # Reverse map for easy lookup (lowercase relative path -> display name)
        filename_to_name = {v.lower(): k for k, v in KNOWN_CORPORA_MAP.items()}
        known_base_names = {os.path.splitext(v.lower())[0]: k for k, v in KNOWN_CORPORA_MAP.items()}
        known_file_names = {os.path.basename(v.lower()): k for k, v in KNOWN_CORPORA_MAP.items()}
        
        # Track display names already assigned to avoid any duplicate listings
        assigned_paths = set()

        # Recursive walk
        count = 0
        for root, dirs, files in os.walk(CORPORA_DIR):
            # Skip unwanted or nested duplicate folders
            for skip_dir in ["BROWN FILES", "corpora", "__pycache__", ".git"]:
                if skip_dir in dirs:
                    dirs.remove(skip_dir)
                
            for file_name in files:
                if file_name.lower().endswith(('.xml', '.txt', '.csv', '.xlsx', '.db', '.duckdb', '.textgrid')):
                    full_path = os.path.join(root, file_name)
                    
                    # Get relative path from CORPORA_DIR
                    rel_path = os.path.relpath(full_path, CORPORA_DIR)
                    rel_path_normalized = rel_path.replace('\\', '/')
                    if rel_path_normalized.lower().startswith('corpora/'):
                        rel_path_normalized = rel_path_normalized[8:]
                        
                    rel_path_lower = rel_path_normalized.lower()
                    fname_lower = file_name.lower()
                    
                    f.write(f"Found: {rel_path_normalized}\n")
                    
                    # 1. Match exact relative path (e.g. "english/bawe.xml")
                    if rel_path_lower in filename_to_name:
                        display_name = filename_to_name[rel_path_lower]
                        available[display_name] = rel_path_normalized
                        assigned_paths.add(full_path.lower())
                    # 2. Match filename (e.g. "bawe.xml")
                    elif fname_lower in known_file_names:
                        display_name = known_file_names[fname_lower]
                        if display_name not in available:
                            available[display_name] = rel_path_normalized
                        assigned_paths.add(full_path.lower())
                    else:
                        rel_base = os.path.splitext(rel_path_lower)[0]
                        fname_base = os.path.splitext(fname_lower)[0]
                        if rel_base in known_base_names or fname_base in known_base_names:
                            display_name = known_base_names.get(rel_base) or known_base_names.get(fname_base)
                            if display_name not in available:
                                available[display_name] = rel_path_normalized
                            assigned_paths.add(full_path.lower())
                        else:
                            # Use relative path as display name for unknown files
                            if root == CORPORA_DIR:
                                available[file_name] = file_name
                            else:
                                available[rel_path_normalized] = rel_path_normalized
                    count += 1
        f.write(f"Total corpora found: {count}\n")

        # Ensure ALL known corpora are listed so they can be loaded/downloaded on-demand
        for display_name, rel_path in KNOWN_CORPORA_MAP.items():
            if display_name not in available:
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

    "BROWN":
        """
        The Brown Corpus, the first million-word electronic corpus of English. It is provided in a **TreeTagger-style XML format** containing token, POS, and lemma.
        <br><br>
        **Source/Citation:** Francis, W. N., & Kucera, H. (1979). **Brown Corpus Manual: Standard Corpus of Present-Day Edited American English for Use with Digital Computers.** Brown University.
        """,
    "KOSLAT-ID (XML Tagged)":
        """
        KOSLAT-ID v.1.0 is the first narrative-annotated corpus of reviews of healthcare facilities in Indonesia. It is provided in a **tagged XML format** (token, POS, lemma).
        <br><br>
        **Source/Citation:** Prihantoro., Yuliawati, S., Ekawati, D., & Rachmat, A. (2026-in press). **KOSLAT-ID v.1.0: The first narrative-annotated corpus of reviews of healthcare facilities in Indonesia.** [Corpora, 21(1), xx-xx.](https://www.prihantoro.com)
        """,
    "BAWE sample (English)":
        """
        The **British Academic Written English (BAWE)** corpus contains proficient undergraduate and master's level writing in various disciplines. This sample includes academic essays and reports.
        <br><br>
        **Source:** Nesi, H., Gardner, S., Thompson, P. & Wickens, P. (2008). **British Academic Written English corpus.** Coventry University.
        """,
    "JP-DICO-JALF":
        """
        The **DICO-JALF Corpus** is a Japanese language learner corpus.
        """,
    "ITA3 Spoken Sample":
        """
        A **multimodal spoken corpus demo** showing integration with Praat `.TextGrid` files and `.wav` audio. 
        <br><br>
        This demo parses TextGrid tiers (like `TA - words`, `TA - phones`, `syllables`, and `ToneUnit`) and automatically links them with acoustic features extracted natively from the audio file using Parselmouth!
        <br><br>
        **Use this to test queries on:** `duration`, `syllable_count`, `f0_mean` (pitch), `f1_mean` (formants), and `pitch_pattern`.
        """,
    "ICNALE Written English":
        """
        The **International Corpus Network of Asian Learners of English (ICNALE)** Written English corpus.
        <br><br>
        **Source/Citation:** Ishikawa, S. (2023). **The ICNALE Guide: An Introduction to a Learner Corpus Study on Asian Learners' L2 English.** Routledge. [Link](https://www.taylorfrancis.com/books/mono/10.4324/9781003252528/icnale-guide-shin-ichiro-ishikawa)
        """,
    "sample of ELAN file (Indonesian)":
        """
        A **sample ELAN language documentation corpus** containing 30 annotated sentences in Indonesian.
        <br><br>
        **Includes 10 parallel annotation layers**: Orthography (`ORT-F`), Delineated Orthography (`ORT-D`), Phonetic (`PHN-F`), Delineated Phonetic (`PHN-D`), Morphemic Gloss (`GLOSS`), Speaker Sex (`SEX`), Location (`LOCATION`), First Language (`FIRST_LANGUAGE`), POS, and Lemma.
        """,
    "sample of ELAN file (Korean)":
        """
        A **sample ELAN language documentation corpus** containing 3 annotated sentences in Korean.
        <br><br>
        **Includes 7 parallel annotation layers**: Hangul (`hangul`), Hangul Delineated (`hangul-delineated`), Transliteration Full (`transliteration-full`), Transliteration Delineated (`transliteration-delineated`), Word Translation (`word-translation`), Morphemic Gloss (`morpheme`), and Free Translation (`free-translation`).
        """,
    "TUFS2023KOMSHI314 (UAM XML)":
        """
        An **Indonesian learner corpus annotated with UAM Corpus Tool XML format**.
        <br><br>
        Showcases **UAM Stand-off phrase tag inheritance** across nested error annotations (`yg0`, `nom0`, `vrbs`, `prep0`).
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
    "Slovenian": "sl",
    "Malagasy": "mg",
    "Albanian": "sq",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Czech": "cs",
    "Estonian": "et",
    "Galician": "gl",
    "Hungarian": "hu",
    "Latin": "la",
    "Mongolian": "mn",
    "Persian": "fa",
    "Slovak": "sk",
    "Swahili": "sw"
}
