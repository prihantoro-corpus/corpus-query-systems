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
        nlp = spacy.load(model_name)
        _SPACY_PIPELINES[lang_code] = nlp
        return nlp
    except (OSError, ImportError):
        try:
            print(f"SpaCy model '{model_name}' not found. Downloading dynamically...")
            import spacy.cli
            spacy.cli.download(model_name)
            import spacy
            nlp = spacy.load(model_name)
            _SPACY_PIPELINES[lang_code] = nlp
            return nlp
        except Exception as e:
            print(f"Failed to download/load SpaCy model '{model_name}': {e}")
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
    try:
        stanza.download(lang_code, verbose=True)
        try:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,mwt,pos,lemma')
        except Exception:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,pos,lemma')
        _STANZA_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as e:
        error_msg = f"Error initializing Stanza for {lang_code}: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

def tag_text_with_stanza(text, lang_code):
    """
    Process text. Tries SpaCy first for massive speedups, falls back to Stanza if unavailable.
    Returns a tuple (list of dicts, error_msg)
    """
    # 1. Try SpaCy first
    spacy_results, spacy_err = tag_text_with_spacy(text, lang_code)
    if spacy_results is not None:
        print(f"Text tagged successfully using SpaCy for '{lang_code}'.")
        return spacy_results, None
        
    print(f"SpaCy not available/failed for '{lang_code}' (Error: {spacy_err}). Falling back to Stanza...")
    
    # 2. Stanza Fallback
    try:
        nlp = get_stanza_pipeline(lang_code)
        if not nlp:
            return tag_text_simple_fallback(text)
            
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
        return results, None
    except Exception as e:
        # Log error for debugging
        try:
            with open("stanza_error.log", "a") as f:
                f.write(f"Stanza error for {lang_code}: {str(e)}\n")
        except:
            pass
        results, _ = tag_text_simple_fallback(text)
        return results, str(e)

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
