import stanza
import logging

# Configure logging
logging.getLogger('stanza').setLevel(logging.WARNING)

# Cache for Stanza pipelines to avoid reloading
_STANZA_PIPELINES = {}
_SPACY_PIPELINES = {}

# Map standard language codes to small/fast SpaCy models
SPACY_MODEL_MAP = {
    'en': 'en_core_web_sm',
    'id': 'id_core_news_sm',
    'zh': 'zh_core_web_sm',
    'ja': 'ja_core_news_sm',
    'ko': 'ko_core_news_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm',
    'es': 'es_core_news_sm',
    'it': 'it_core_news_sm',
    'ru': 'ru_core_news_sm',
    'pt': 'pt_core_news_sm',
    'nl': 'nl_core_news_sm',
    'el': 'el_core_news_sm',
    'pl': 'pl_core_news_sm',
    'uk': 'uk_core_news_sm',
    'ro': 'ro_core_news_sm',
    'sv': 'sv_core_news_sm',
    'da': 'da_core_news_sm',
    'nb': 'nb_core_news_sm',
    'fi': 'fi_core_news_sm',
    'ca': 'ca_core_news_sm',
    'hr': 'hr_core_news_sm',
    'lt': 'lt_core_news_sm',
    'mk': 'mk_core_news_sm',
    'sl': 'sl_core_news_sm',
}

def get_spacy_pipeline(lang_code):
    """
    Get or dynamically download a SpaCy pipeline for the specified language.
    """
    global _SPACY_PIPELINES
    if lang_code in _SPACY_PIPELINES:
        return _SPACY_PIPELINES[lang_code]
    
    if lang_code not in SPACY_MODEL_MAP:
        return None
        
    model_name = SPACY_MODEL_MAP[lang_code]
    try:
        import spacy
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"SpaCy model '{model_name}' not found locally. Downloading on demand...")
            from spacy.cli import download
            download(model_name)
            nlp = spacy.load(model_name)
        _SPACY_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as e:
        print(f"SpaCy model '{model_name}' download or load failed: {e}")
        return None

def tag_text_with_spacy(text, lang_code):
    """
    Process text with SpaCy.
    """
    try:
        nlp = get_spacy_pipeline(lang_code)
        if not nlp:
            return None, "SpaCy model mapping not available or download failed"
            
        # Handle large text inputs safely
        if len(text) > nlp.max_length:
            nlp.max_length = len(text) + 1000
            
        doc = nlp(text)
        results = []
        sent_id = 0
        for sent in doc.sents:
            sent_id += 1
            for token in sent:
                if token.is_space:
                    continue
                results.append({
                    'token': token.text,
                    'pos': token.pos_,  # Returns Universal Part of Speech tags (UPOS)
                    'lemma': token.lemma_ if token.lemma_ else token.text,
                    'sent_id': sent_id,
                    'ent_type': token.ent_type_ if token.ent_type_ else ""
                })
        return results, None
    except Exception as e:
        return None, str(e)

def split_sentences_custom(text):
    r"""
    Splits text into sentences using regex, preserving closing quotes/parentheses.
    """
    import re
    # Matches everything up to punctuation, followed by optional closing quotes/parens, 
    # and then space or end of string. If no punctuation is found, matches the rest (.+)
    pattern = r'.*?[\.\?\!](?:[\s]*[\'\"’”\)\]]+)*(?:\s+|$)|.+'
    sentences = re.findall(pattern, text, flags=re.DOTALL)
    
    return [s.strip() for s in sentences if s.strip()]

def get_stanza_pipeline(lang_code):
    """
    Get or create a Stanza pipeline for the specified language.
    Attempts local loading first to remain completely offline, falling back to download if missing.
    """
    global _STANZA_PIPELINES
    
    if lang_code in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[lang_code]
    
    # 1. Attempt to load offline directly
    try:
        print(f"Initializing Stanza pipeline offline for '{lang_code}'...")
        try:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,mwt,pos,lemma', download_method=None)
        except Exception:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,pos,lemma', download_method=None)
        _STANZA_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as local_err:
        print(f"Local Stanza load failed for '{lang_code}' ({local_err}). Attempting download...")
        
    # 2. Fallback to download if local load fails
    print(f"Skipping dynamic Stanza download on cloud for '{lang_code}'.")
    return None

import subprocess
import tempfile
import os
import platform

# Map stanza lang codes to treetagger parameter files
TREETAGGER_LANG_MAP = {
    'id': 'indonesian/indonesian_v311225.par',
    'mg': 'malagasy/malagasy.par', # Replace with actual if name is different
    'en': 'english/english.par'
}

def tag_text_with_treetagger(text, lang_code):
    """
    Process text with TreeTagger. Detects OS and uses correct binary.
    Returns (list of dicts, error_msg)
    """
    if lang_code not in TREETAGGER_LANG_MAP:
        return None, f"Language '{lang_code}' not supported by local TreeTagger."

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tt_dir = os.path.join(base_dir, 'treetagger')
    
    # OS Detection
    system = platform.system().lower()
    if system == 'windows':
        executable = os.path.join(tt_dir, '_Windows', 'tree-tagger.exe')
    else:
        executable = os.path.join(tt_dir, '_Linux', 'tree-tagger')
        
    param_file = os.path.join(tt_dir, 'language_model', TREETAGGER_LANG_MAP[lang_code].replace('/', os.sep))
    
    # Look for any .par file in the directory if the specific one is not found
    if not os.path.exists(param_file):
        lang_folder = os.path.join(tt_dir, 'language_model', TREETAGGER_LANG_MAP[lang_code].split('/')[0])
        if os.path.exists(lang_folder):
            for f in os.listdir(lang_folder):
                if f.endswith('.par'):
                    param_file = os.path.join(lang_folder, f)
                    break
    
    if not os.path.exists(executable):
        return None, f"TreeTagger executable not found at {executable}."
    if not os.path.exists(param_file):
        return None, f"TreeTagger parameter file not found at {param_file}."
        
    # Check if executable can be run (Linux might need chmod +x, or wrong arch)
    if system != 'windows' and not os.access(executable, os.X_OK):
        try:
            os.chmod(executable, 0o755)
        except Exception:
            pass

    try:
        # Tokenize text safely first
        sentences_tokens = tokenize_text_only(text, lang_code)
        
        # Apply Indonesian Clitic Splitter if ID
        if lang_code == 'id':
            from core.preprocessing.indonesian_tokenizer import tokenize_indonesian_clitics
            processed_sentences = []
            for sent in sentences_tokens:
                processed_sentences.append(tokenize_indonesian_clitics(sent))
            sentences_tokens = processed_sentences
            
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f_in:
            for sent in sentences_tokens:
                for token in sent:
                    if token.strip():
                        f_in.write(token.strip() + "\n")
            input_path = f_in.name
            
        with tempfile.NamedTemporaryFile(mode='r', encoding='utf-8', delete=False) as f_out:
            output_path = f_out.name

        cmd = [executable, param_file, input_path, output_path, '-token', '-lemma']
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Read output
        with open(output_path, 'r', encoding='utf-8') as f:
            output_text = f.read()
            
        os.remove(input_path)
        os.remove(output_path)
        
        # Parse vertical output: token \t pos \t lemma
        results = []
        sent_id = 1
        lines = [l.strip() for l in output_text.split('\n') if l.strip()]
        
        for line in lines:
            parts = line.split('\t')
            if not parts or not parts[0]:
                continue
            token = parts[0]
            pos = parts[1] if len(parts) > 1 else 'TAG'
            lemma = parts[2] if len(parts) > 2 else token
            
            results.append({
                'token': token,
                'pos': pos,
                'lemma': lemma,
                'sent_id': sent_id,
                'ent_type': ""
            })
            
            # Simple sentence splitting heuristic based on punctuation
            if token in ['.', '!', '?']:
                sent_id += 1
                
        return results, None
        
    except subprocess.CalledProcessError as e:
        return None, f"TreeTagger execution failed: {e.stderr}"
    except Exception as e:
        return None, f"TreeTagger error: {str(e)}"

def tag_text_with_stanza(text, lang_code):
    """
    Process text. Tries TreeTagger first, then SpaCy, then Stanza.
    Returns a tuple (list of dicts, error_msg)
    """
    # 1. Try TreeTagger first (Highest Priority)
    tt_results, tt_err = tag_text_with_treetagger(text, lang_code)
    if tt_results is not None:
        print(f"Text tagged successfully using TreeTagger for '{lang_code}'.")
        return tt_results, None
        
    print(f"TreeTagger not available/failed for '{lang_code}' (Error: {tt_err}). Falling back to SpaCy/Stanza...")

    # 2. Try SpaCy
    spacy_results, spacy_err = tag_text_with_spacy(text, lang_code)
    if spacy_results is not None:
        print(f"Text tagged successfully using SpaCy for '{lang_code}'.")
        # If TreeTagger failed but was attempted, we could pass the warning. 
        # But we must return the result. We'll return the error string as warning.
        return spacy_results, f"treetagger fail, switching to SpaCy. {tt_err}"
        
    print(f"SpaCy not available/failed for '{lang_code}' (Error: {spacy_err}). Falling back to Stanza...")
    
    # 3. Try Stanza
    try:
        nlp = get_stanza_pipeline(lang_code)
        if nlp:
            # Use Stanza's native sentence splitting for better results
            doc = nlp(text)
            
            results = []
            for sent_id, stanza_sent in enumerate(doc.sentences, 1):
                for word in stanza_sent.words:
                    results.append({
                        'token': word.text,
                        'pos': word.upos, 
                        'lemma': word.lemma if word.lemma else word.text,
                        'sent_id': sent_id,
                        'ent_type': ""
                    })
            print(f"Text tagged successfully using Stanza for '{lang_code}'.")
            return results, f"treetagger fail, switching to Stanza. {tt_err}" if tt_err else None
    except Exception as e:
        print(f"Stanza error for {lang_code}: {str(e)}")
        
    # 3. Try Custom PKL Model as Last Resort if Stanza fails/is disabled
    import os
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model', f'{lang_code}-hmm.pkl')
    if os.path.exists(pkl_path):
        print(f"Found custom {lang_code}-hmm.pkl model. Using it as last resort fallback...")
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                custom_tagger = pickle.load(f)
            sentences_tokens = tokenize_text_only(text, lang_code)
            results = []
            sent_id = 0
            for sent_tokens in sentences_tokens:
                sent_id += 1
                tagged_tokens = custom_tagger.tag(sent_tokens)
                for i, token_str in enumerate(sent_tokens):
                    item = tagged_tokens[i]
                    results.append({
                        'token': token_str,
                        'pos': item['pos'],
                        'lemma': item['lemma'],
                        'sent_id': sent_id,
                        'ent_type': ""
                    })
            return results, None
        except Exception as e:
            print(f"Failed to load/tag with {lang_code}-hmm.pkl: {e}")
    # 4. Fallback to Simple Regex Tokenizer
    fallback_res, fallback_err = tag_text_simple_fallback(text)
    return fallback_res, f"treetagger fail, switching to fallback. {tt_err}" if tt_err else fallback_err

def tag_text_simple_fallback(text):
    """
    Fallback tagging. Returns (results, error)
    """
    import re
    
    sentences = split_sentences_custom(text)
    
    results = []
    sent_id = 0
    
    for sent_text in sentences:
        sent_id += 1
        # Simple tokenization: split but preserve punctuation
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
        tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        
        for token in tokens:
            results.append({
                'token': token,
                'pos': 'TAG', 
                'lemma': token,
                'sent_id': sent_id,
                'ent_type': ""
            })
            
    return results, None

def get_stanza_tokenizer_only(lang_code):
    global _STANZA_PIPELINES
    key = f"{lang_code}_tokenize_only"
    if key in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[key]
        
    try:
        nlp = stanza.Pipeline(lang=lang_code, processors='tokenize', download_method=None)
    except Exception:
        print(f"Skipping dynamic Stanza tokenizer download on cloud for '{lang_code}'.")
        return None
    _STANZA_PIPELINES[key] = nlp
    return nlp

def tokenize_text_with_spacy(text, lang_code):
    nlp = get_spacy_pipeline(lang_code)
    if not nlp:
        return None
    try:
        # We must add sentencizer if parser is disabled, otherwise doc.sents fails
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        
        # Handle large text inputs safely
        if len(text) > nlp.max_length:
            nlp.max_length = len(text) + 1000
            
        # Disable all parsing, tagging, lemmatization to run tokenizer only
        with nlp.select_pipes(disable=['tagger', 'parser', 'ner', 'lemmatizer']):
            doc = nlp(text)
            sentences = []
            for sent in doc.sents:
                sent_tokens = [token.text for token in sent if not token.is_space]
                if sent_tokens:
                    sentences.append(sent_tokens)
            return sentences
    except Exception:
        return None

def tokenize_text_with_stanza(text, lang_code):
    nlp = get_stanza_tokenizer_only(lang_code)
    if not nlp:
        return None
    try:
        doc = nlp(text)
        sentences = []
        for stanza_sent in doc.sentences:
            sent_tokens = [word.text for word in stanza_sent.words]
            if sent_tokens:
                sentences.append(sent_tokens)
        return sentences
    except Exception:
        return None

def tokenize_text_only(text, lang_code):
    """
    Splits text into sentences and tokens, bypassing tagging & parsing.
    Returns: list of list of str (list of sentences, where each sentence is a list of tokens)
    """
    if not lang_code or lang_code == "OTHER":
        import re
        sentences = split_sentences_custom(text)
        results = []
        for sent_text in sentences:
            cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
            tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
            if tokens:
                results.append(tokens)
        return results
        
    # 1. Try SpaCy
    spacy_res = tokenize_text_with_spacy(text, lang_code)
    if spacy_res is not None:
        return spacy_res
        
    # 2. Try Stanza
    stanza_res = tokenize_text_with_stanza(text, lang_code)
    if stanza_res is not None:
        return stanza_res
        
    # 3. Fallback
    import re
    sentences = split_sentences_custom(text)
    results = []
    for sent_text in sentences:
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', sent_text)
        tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        if tokens:
            results.append(tokens)
    return results

