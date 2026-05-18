import stanza
import logging

# Configure logging
logging.getLogger('stanza').setLevel(logging.WARNING)

# Cache for Stanza pipelines to avoid reloading
_STANZA_PIPELINES = {}

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
    Downloads the model if not present.
    """
    global _STANZA_PIPELINES
    
    if lang_code in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[lang_code]
    
    try:
        print(f"Initializing Stanza pipeline for '{lang_code}'...")
        stanza.download(lang_code, verbose=True)
        
        # MWT is not supported by all languages (e.g., English, Indonesian, Chinese)
        # We try with MWT, and if it fails, we try without it.
        try:
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,mwt,pos,lemma')
        except Exception:
            print(f"MWT processor not supported for {lang_code}. Retrying without MWT...")
            nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,pos,lemma')
            
        _STANZA_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as e:
        error_msg = f"Error initializing Stanza for {lang_code}: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

def tag_text_with_stanza(text, lang_code):
    """
    Process text with Stanza.
    Returns a tuple (list of dicts, error_msg)
    """
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
                    'sent_id': sent_id
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
                'sent_id': sent_id
            })
            
    return results, None
