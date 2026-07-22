import os

_LEXICON_SET = None

def _load_lexicon():
    global _LEXICON_SET
    if _LEXICON_SET is not None:
        return
    _LEXICON_SET = set()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lexicon_path = os.path.join(base_dir, 'model', 'lexicon2.txt')
    
    if os.path.exists(lexicon_path):
        try:
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        word = parts[0].strip().lower()
                        if word:
                            _LEXICON_SET.add(word)
        except Exception as e:
            print(f"Failed to load lexicon: {e}")

def _is_valid_base(word):
    """Check if the word is in lexicon or passes heuristics."""
    if not _LEXICON_SET:
        _load_lexicon()
        
    lower_word = word.lower()
    
    # If in lexicon, it's valid
    if _LEXICON_SET and lower_word in _LEXICON_SET:
        return True
        
    # Heuristic for unknown words (e.g. Names)
    # If it starts with a capital letter and length >= 3, we allow it (Named Entity)
    if word and word[0].isupper() and len(word) >= 3:
        return True
        
    return False

def tokenize_indonesian_clitics(tokens):
    """
    Takes a list of pre-tokenized words and splits Indonesian clitics 
    (ku-, kau-, -nya, -ku, -mu, -lah, -kah, -pun) safely using dictionary checks.
    """
    _load_lexicon()
    
    result = []
    
    for token in tokens:
        # Ignore very short tokens or punctuation
        if len(token) <= 3 or not any(c.isalpha() for c in token):
            result.append(token)
            continue
            
        current_word = token
        lower_word = current_word.lower()
        
        # PROKLITIK: ku-, kau-
        prefix = None
        if lower_word.startswith("ku") and len(lower_word) >= 4: # e.g. kubawa (6) base bawa
            base = current_word[2:]
            if _is_valid_base(base):
                prefix = current_word[:2] + "-"
                current_word = base
                lower_word = current_word.lower()
        elif lower_word.startswith("kau") and len(lower_word) >= 5: # e.g. kaubawa (7) base bawa
            base = current_word[3:]
            if _is_valid_base(base):
                prefix = current_word[:3] + "-"
                current_word = base
                lower_word = current_word.lower()
                
        # ENKLITIK: 
        suffixes = []
        
        # Check particle first (-lah, -kah, -pun)
        for part in ["lah", "kah", "pun"]:
            if lower_word.endswith(part) and len(lower_word) >= len(part) + 2:
                base = current_word[:-len(part)]
                if _is_valid_base(base) or (
                    # Allow cascading if it ends with another clitic (e.g. -nya)
                    base.lower().endswith("nya") or base.lower().endswith("ku") or base.lower().endswith("mu")
                ):
                    suffixes.insert(0, "-" + current_word[-len(part):])
                    current_word = base
                    lower_word = current_word.lower()
                    break # Only one particle at the end
                    
        # Check pronominal enclitic (-nya, -ku, -mu)
        if lower_word.endswith("nya") and len(lower_word) >= 5: # e.g. esnya (5)
            base = current_word[:-3]
            if _is_valid_base(base):
                suffixes.insert(0, "-" + current_word[-3:])
                current_word = base
                lower_word = current_word.lower()
        elif (lower_word.endswith("ku") or lower_word.endswith("mu")) and len(lower_word) >= 4: # e.g. tasku (5)
            base = current_word[:-2]
            if _is_valid_base(base):
                suffixes.insert(0, "-" + current_word[-2:])
                current_word = base
                lower_word = current_word.lower()
                
        # Reconstruct sequence
        if prefix:
            result.append(prefix)
        result.append(current_word)
        result.extend(suffixes)
        
    return result
