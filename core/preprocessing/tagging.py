import stanza
import logging

# Configure logging
logging.getLogger('stanza').setLevel(logging.WARNING)

# Cache for Stanza pipelines to avoid reloading
_STANZA_PIPELINES = {}

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
        
        # Simulating the standalone script approach:
        # Just call download first (it checks if exists internally)
        stanza.download(lang_code, verbose=True)
        
        # Then init pipeline
        nlp = stanza.Pipeline(lang=lang_code, processors='tokenize,mwt,pos,lemma')
        
        _STANZA_PIPELINES[lang_code] = nlp
        return nlp
    except Exception as e:
        error_msg = f"Error initializing Stanza for {lang_code}: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

def tag_text_with_stanza(text, lang_code):
    """
    Process text with Stanza.
    Returns a list of dicts: {'token': x, 'pos': y, 'lemma': z, 'sent_id': n}
    """
    nlp = get_stanza_pipeline(lang_code)
    if not nlp:
        # If stanza fails, fallback to simple
        return tag_text_simple_fallback(text)
        
    doc = nlp(text)
    
    results = []
    sent_id = 0
    
    for sentence in doc.sentences:
        sent_id += 1
        for word in sentence.words:
            results.append({
                'token': word.text,
                'pos': word.upos, # Universal POS tags
                'lemma': word.lemma if word.lemma else word.text,
                'sent_id': sent_id
            })
            
    return results

def tag_text_simple_fallback(text):
    """
    Fallback tagging for unsupported languages or when Stanza fails.
    Uses 'TAG' for POS and Token for Lemma.
    """
    import re
    
    # Simple tokenization: split by whitespace and keep punctuation
    # A regex that splits but keeps delimiters could be better
    # or just simple whitespace for now as requested "text is untagged"
    
    # "cleaned_text" logic from corpus_loader was:
    # re.sub(r'([^\w\s])', r' \1 ', raw_text)
    # tokens = [t.strip() for t in cleaned_text.split() if t.strip()] 
    
    cleaned_text = re.sub(r'([^\w\s])', r' \1 ', text)
    tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
    
    results = []
    # All one sentence? Or try to split newlines? 
    # Let's treat it as one "stream" but maybe increment sent_id on newlines if we had line info.
    # For now, just one dummy sentence ID 1, or maybe map to lines?
    # Simpler to just map all to 0 or 1 if we don't do sentence splitting.
    
    for token in tokens:
        results.append({
            'token': token,
            'pos': '##', # Per user request: "When other is chosen, instead of ##, use 'TAG' instead." -> WAIT, User said: "use 'TAG' instead"
            # User said: "When other is chosen, instead of ##, use 'TAG' instead."
            # My logic in corpus_loader used "##". I should use "TAG".
            'lemma': token,
            'sent_id': 1
        })
        
    # Correction: Update 'pos' to 'TAG' based on request, but wait- 
    # user said "For no, only ENglish, Indonesia and Japan is available... When other is chosen, instead of ##, use 'TAG' instead."
    # So:
    for r in results:
        r['pos'] = 'TAG'
        
    return results
