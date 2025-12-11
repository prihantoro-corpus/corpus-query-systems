# app.py
# CORTEX -- Corpus Explorer Version Alpha (10-Dec-25)

# (All imports and constant definitions remain the same as the Third Fix)
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from io import BytesIO, StringIO 
import tempfile 
import os 
import re 
import requests
import matplotlib.pyplot as plt 
# (imports continue...)
try: from wordcloud import WordCloud; WORDCLOUD_FEATURE_AVAILABLE = True
except ImportError: WORDCLOUD_FEATURE_AVAILABLE = False
try: from pyvis.network import Network; PYVIS_FEATURE_AVAILABLE = True
except ImportError: PYVIS_FEATURE_AVAILABLE = False
try: from cefrpy import CEFRAnalyzer; CEFR_ANALYZER = CEFRAnalyzer(); CEFR_FEATURE_AVAILABLE = True
except ImportError: CEFR_FEATURE_AVAILABLE = False
try: import eng_to_ipa as ipa; IPA_FEATURE_AVAILABLE = True
except ImportError: IPA_FEATURE_AVAILABLE = False
import streamlit.components.v1 as components 
import xml.etree.ElementTree as ET # Import for XML parsing


st.set_page_config(page_title="CORTEX -- Corpus Explorer Version Alpha (10-Dec-25) by PRIHANTORO (www.prihantoro.com; prihantoro@live.undip.ac.id)", layout="wide") 

# (Constants and Session State initialization remain the same)
KWIC_MAX_DISPLAY_LINES = 100
# ... (rest of constants)

# Define global names for parallel mode
SOURCE_LANG_CODE = 'EN'
TARGET_LANG_CODE = 'ID'
DEFAULT_LANG_CODE = 'RAW'

# ---------------------------
# Initializing Session State
# (All session state initializations remain the same, including the critical ones)
if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
if 'corpus_loaded' not in st.session_state: st.session_state['corpus_loaded'] = False
# (rest of session state initialization)
if 'view' not in st.session_state: st.session_state['view'] = 'overview'
if 'trigger_analyze' not in st.session_state: st.session_state['trigger_analyze'] = False
if 'n_gram_results_df' not in st.session_state: st.session_state['n_gram_results_df'] = pd.DataFrame()
if 'parallel_mode' not in st.session_state: st.session_state['parallel_mode'] = False
if 'target_sent_map' not in st.session_state: st.session_state['target_sent_map'] = {}
if 'monolingual_xml_file_upload' not in st.session_state: st.session_state['monolingual_xml_file_upload'] = None
if 'xml_structure_error' not in st.session_state: st.session_state['xml_structure_error'] = None
if 'user_explicit_lang_code' not in st.session_state: st.session_state['user_explicit_lang_code'] = 'EN'
if 'last_built_in_content' not in st.session_state: st.session_state['last_built_in_content'] = None


BUILT_IN_CORPORA = {
    "Select built-in corpus...": None,
    "BROWN (EN XML Tagged)": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/BrownCorpus.xml",
    "KOSLAT (ID XML Tagged)": "https://raw.githubusercontent.com/prihantoro-corpus/corpus-query-systems/main/KOSLAT-full.xml",
}
# (POS_COLOR_MAP, PUNCTUATION, utility functions like pmw_to_zipf, reset_analysis, generate_kwic, etc., remain the same)
# ...

# --- Built-in Download (CACHED) ---
@st.cache_data
def get_corpus_data_from_url(corpus_url):
    """Downloads a file and caches the raw binary content (robustly)."""
    try:
        response = requests.get(corpus_url, stream=True)
        response.raise_for_status() 
        return response.content 
    except Exception as e:
        # NOTE: This error is explicitly caught and displayed in handle_corpus_loading
        st.session_state['xml_structure_error'] = f"Download/Network Error: {e}" 
        return None
# ----------------------------------

# --- Core XML Parser (non-cached, remains same structure) ---
def parse_xml_content_to_df(file_source):
    """
    Core parser, wraps ET logic with sanitization.
    (Parser logic omitted for brevity, identical to Fix 3/4)
    """
    cleaned_xml_content = sanitize_xml_content(file_source)
    if cleaned_xml_content is None: return None
    
    # ... (XML parsing logic remains the same)
    try:
        root = ET.fromstring(cleaned_xml_content)
        # ... (language detection)
        # ... (token/sentence extraction)
        df_data = [] # ...
        sent_map = {} # ...
        # ... (rest of parsing logic)
        
    except Exception as e:
        # CRITICAL: We catch the final parsing error here
        file_name_label = getattr(file_source, 'name', 'Uploaded XML File')
        st.session_state['xml_structure_error'] = f"XML PARSING FAILURE (File: {file_name_label}): {e}"
        return None
        
    # (Final return structure of {'lang_code', 'df_data', 'sent_map'})
    # Since parsing logic is verbose, the full function is omitted but assumed correct.
    
    # Placeholder for actual return after successful parsing:
    if not df_data:
        file_name_label = getattr(file_source, 'name', 'Uploaded XML File')
        st.session_state['xml_structure_error'] = f"XML PARSING SUCCESSFUL, but no tokenized data extracted from file: {file_name_label}."
        return None
    
    return {'lang_code': lang_code, 'df_data': df_data, 'sent_map': sent_map}
# -----------------------------------------------------

# --- Loading functions (NO CACHE, return DF) ---
# (load_monolingual_corpus_files, load_xml_parallel_corpus, load_excel_parallel_corpus_file, load_corpus_file_built_in remain structurally the same, but without @st.cache_data and relying on globals/returns)
# ...
def load_monolingual_corpus_files(file_sources, explicit_lang_code, selected_format):
    global SOURCE_LANG_CODE, TARGET_LANG_CODE
    # ... (loading logic)
    df_src = pd.DataFrame(all_df_data)
    df_src["_token_low"] = df_src["token"].str.lower()
    # ... (structure extraction)
    return df_src
# ... (other load functions return df_src)


# -----------------------------------------------------
# MASTER LOADING HANDLER (NEW - CRITICAL ERROR TRAPPING)
# -----------------------------------------------------

def handle_corpus_loading():
    """
    Executes the appropriate loading function based on the sidebar selection
    and updates the session state only if successful.
    """
    selected_corpus_name = st.session_state.get('corpus_select')
    corpus_mode = st.session_state.get('corpus_mode_radio')
    explicit_lang_code = st.session_state.get('user_explicit_lang_code')
    
    df_result = None
    corpus_loaded_name = ""
    st.session_state['xml_structure_error'] = None # Clear prior errors
    
    st.session_state['processed_df'] = None
    st.session_state['corpus_loaded'] = False

    try:
        # 1. BUILT-IN CORPUS
        if selected_corpus_name != "Select built-in corpus...":
            corpus_url = BUILT_IN_CORPORA[selected_corpus_name] 
            with st.spinner(f"Downloading {selected_corpus_name} via network..."):
                raw_content = get_corpus_data_from_url(corpus_url)
                st.session_state['last_built_in_content'] = raw_content
                
            if raw_content is not None:
                with st.spinner(f"Processing {selected_corpus_name}..."):
                    df_result = load_corpus_file_built_in(raw_content, selected_corpus_name, explicit_lang_code)
                    if df_result is not None:
                        corpus_loaded_name = selected_corpus_name
            # --- Check specific download failure that bypasses cache ---
            elif st.session_state.get('xml_structure_error') and "Download/Network Error" in st.session_state['xml_structure_error']:
                 # If download failed, we exit the try block and the failure is handled below
                 pass 
            
        # 2. CUSTOM UPLOAD (MONOLINGUAL)
        elif corpus_mode == "Monolingual Corpus" and st.session_state.get('mono_file_upload'):
             uploaded_files_mono = st.session_state['mono_file_upload']
             selected_format_mono = st.session_state.get('mono_format_select')
             
             with st.spinner(f"Processing Monolingual Corpus ({len(uploaded_files_mono)} file(s))..."):
                 df_result = load_monolingual_corpus_files(uploaded_files_mono, explicit_lang_code, selected_format_mono)
                 if df_result is not None:
                     corpus_loaded_name = f"Monolingual ({SOURCE_LANG_CODE}, {selected_format_mono})"
                     
        # 3. CUSTOM UPLOAD (PARALLEL)
        elif corpus_mode == "Parallel Corpus":
            parallel_file_mode = st.session_state.get('parallel_file_mode_radio')
            
            if parallel_file_mode == "One corpus file" and st.session_state.get('parallel_excel_file_upload'):
                parallel_excel_file = st.session_state['parallel_excel_file_upload']
                excel_format = st.session_state.get('excel_format_radio')
                with st.spinner("Processing Excel Parallel Corpus..."):
                    df_result = load_excel_parallel_corpus_file(parallel_excel_file, excel_format)
                    if df_result is not None:
                        corpus_loaded_name = f"Parallel (Excel) ({SOURCE_LANG_CODE}/{TARGET_LANG_CODE})"
                        st.session_state['parallel_mode'] = True

            elif parallel_file_mode == "Two corpus files (aligned IDs required)" and st.session_state.get('xml_src_file_upload') and st.session_state.get('xml_tgt_file_upload'):
                xml_src_file = st.session_state['xml_src_file_upload']
                xml_tgt_file = st.session_state['xml_tgt_file_upload']
                src_lang_input = st.session_state.get('src_lang_code_input')
                tgt_lang_input = st.session_state.get('tgt_lang_code_input')
                
                with st.spinner("Processing XML Parallel Corpus..."):
                    df_result = load_xml_parallel_corpus(xml_src_file, xml_tgt_file, src_lang_input, tgt_lang_input)
                    if df_result is not None:
                        corpus_loaded_name = f"Parallel (XML) ({SOURCE_LANG_CODE}/{TARGET_LANG_CODE})"
                        st.session_state['parallel_mode'] = True
    
    except Exception as e:
        # Generic catch-all for any unhandled exception during the loading process
        st.session_state['xml_structure_error'] = f"UNHANDLED CRITICAL LOAD ERROR: {type(e).__name__}: {e}"
        print(f"CRITICAL ERROR LOG: {st.session_state['xml_structure_error']}") # Log to console for environment debug
        df_result = None

    
    # --- FINAL STATE UPDATE ---
    if df_result is not None:
        st.session_state['processed_df'] = df_result
        st.session_state['corpus_loaded'] = True
        st.session_state['corpus_name'] = corpus_loaded_name
        st.session_state['view'] = 'overview'
        st.rerun()
        
    elif st.session_state.get('corpus_name'):
         st.session_state['corpus_name'] = None
         # If load failed, and an error message was written to 'xml_structure_error', it will be displayed below.
# -----------------------------------------------------

# (Rest of the script: UI/Sidebar logic, Navigation, Load Check, Module Logic remains the same)
# ...

# --- CHECK FOR LOAD SUCCESS OR FAILURE ---
# This block is essential for displaying the specific error message now stored in xml_structure_error
if not st.session_state['corpus_loaded']:
    
    xml_error = st.session_state.get('xml_structure_error')

    if xml_error:
        st.header(f"❌ Corpus Load Failed (Specific Error)")
        st.markdown("---")
        st.error(f"**Critical Error:** The file processing failed during download or parsing.")
        st.warning(f"**Specific Error Details:** {xml_error}")
        st.info("If this is the Brown Corpus, the URL may be blocked by your host, or the XML format caused a hard parser crash. Please try uploading the file manually.")

    elif st.session_state.get('corpus_name') != "No Corpus Loaded" or st.session_state.get('mono_file_upload') or st.session_state.get('parallel_excel_file_upload') or st.session_state.get('xml_src_file_upload'):
        # Generic error after a failed manual load attempt where no specific XML error was logged
        st.header("❌ Corpus Load Failed")
        st.markdown("---")
        st.error("The corpus could not be loaded. Check your file format, or try clicking 'Load Corpus' again.")

    else:
        # Show welcome screen if nothing is selected or uploaded
        st.header("👋 Welcome to CORTEX!")
        st.markdown("---")
        st.markdown("## Get Started")
        st.markdown("**1. Select/Upload** a corpus in the sidebar.")
        st.markdown("**2. Click '🚀 Load Corpus'** in the sidebar to process the data.")

    st.stop() 

# (Rest of the application modules for Overview, Dictionary, Concordance, Collocation follow)
# ...
