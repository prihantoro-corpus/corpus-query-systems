import os
import math
import string
import duckdb
import pandas as pd
from core.modules.overview import get_corpus_language, get_pos_definitions

def classify_pos_tag(tag, definition, lang="English"):
    """
    Classifies a POS tag into Noun (N), Verb (V), Adjective (Adj), Adverb (Adv),
    or None based on POS definitions and tag prefix fallbacks.
    """
    if not tag:
        return None
        
    tag_clean = tag.strip().lower()
    def_clean = definition.strip().lower() if definition else ""
    
    # 1. Check definition keywords (supports any language if definitions are provided)
    if def_clean:
        # Extract the base definition before parentheses/separators to avoid description collisions
        # e.g., "adverb (words modifying verbs, adjectives...)" -> base is "adverb"
        base_def = def_clean
        for sep in ('(', ',', '-', ';', ':', '/'):
            if sep in base_def:
                base_def = base_def.split(sep)[0]
        base_def = base_def.strip()
        
        # Check adverb first to prevent collision with adjective
        if "adverb" in base_def or "副詞" in base_def:
            return "Adv"
        if "adjective" in base_def or "形容詞" in base_def:
            return "Adj"
        if "noun" in base_def or "名詞" in base_def:
            return "N"
        if "verb" in base_def or "動詞" in base_def:
            return "V"

    # 2. Fall back to standard language-specific prefixes / tag structures
    lang_clean = lang.lower() if lang else "english"
    
    if "japanese" in lang_clean or "jp" in lang_clean:
        # Japanese BCCWJ prefixes
        if tag_clean.startswith("名詞") or "名詞" in tag_clean:
            return "N"
        if tag_clean.startswith("動詞") or "動詞" in tag_clean:
            return "V"
        if tag_clean.startswith("形容詞") or tag_clean.startswith("形状詞") or "形容詞" in tag_clean:
            return "Adj"
        if tag_clean.startswith("副詞") or "副詞" in tag_clean:
            return "Adv"
            
    # Universal Dependencies (UPOS) fallbacks
    if tag_clean in ("noun", "propn", "pron"):
        return "N"
    if tag_clean in ("verb", "aux"):
        return "V"
    if tag_clean == "adj":
        return "Adj"
    if tag_clean == "adv":
        return "Adv"
        
    # Indonesian BPPT / other standard tagset fallbacks
    if tag_clean in ("nsd", "nsm", "prp", "prp$", "pr"):
        return "N"
    if tag_clean in ("vsd", "vbt", "md"):
        return "V"
    if tag_clean == "neg":
        return "Adv"
            
    # Default fallback (English Penn Treebank / CLAWS)
    if tag_clean.startswith("nn") or tag_clean.startswith("np"):
        return "N"
    if tag_clean.startswith("vb") or tag_clean.startswith("vv") or tag_clean.startswith("vh"):
        return "V"
    if tag_clean.startswith("jj"):
        return "Adj"
    if tag_clean.startswith("rb"):
        return "Adv"
        
    return None

def is_punctuation_or_number(token):
    """Checks if a token is purely punctuation or purely numeric."""
    if not token:
        return True
    tok_str = str(token).strip()
    if not tok_str:
        return True
    if all(c in string.punctuation or c.isspace() for c in tok_str):
        return True
    if tok_str.replace(".", "", 1).isdigit():
        return True
    return False

def calculate_mtld(filtered_lemmas, threshold=0.72):
    """
    Calculates the Measure of Textual Lexical Diversity (MTLD) for a list of lemmas.
    """
    N = len(filtered_lemmas)
    if N == 0:
        return 0.0
        
    def mtld_dir(lemmas):
        factor_count = 0
        ttr = 1.0
        seg_len = 0
        seg_types = set()
        
        for word in lemmas:
            seg_len += 1
            seg_types.add(word)
            ttr = len(seg_types) / seg_len
            if ttr < threshold:
                factor_count += 1
                seg_len = 0
                seg_types = set()
                ttr = 1.0
                
        # Handle the last incomplete segment
        if seg_len > 0:
            factor = (1.0 - ttr) / (1.0 - threshold) if ttr < 1.0 else 0.0
            factor_count += factor
            
        return N / factor_count if factor_count > 0 else N

    forward = mtld_dir(filtered_lemmas)
    backward = mtld_dir(list(reversed(filtered_lemmas)))
    
    return (forward + backward) / 2.0

def calculate_generic_complexity(lemmas):
    """
    Computes generic, language-agnostic complexity measures:
    TTR, RTTR, CTTR, LogTTR, Uber Index, MTLD, and STTR/MSTTR at 50, 100, 200 sizes.
    """
    filtered_lemmas = [str(l).lower() for l in lemmas if not is_punctuation_or_number(l)]
    N = len(filtered_lemmas)
    if N == 0:
        return {
            "N": 0, "V": 0, "TTR": 0.0, "RTTR": 0.0, "CTTR": 0.0,
            "LogTTR": 0.0, "Uber": 0.0, "MTLD": 0.0, "STTR_50": 0.0, "STTR_100": 0.0, "STTR_200": 0.0
        }
        
    unique_lemmas = set(filtered_lemmas)
    V = len(unique_lemmas)
    
    ttr = V / N
    rttr = V / math.sqrt(N)
    cttr = V / math.sqrt(2 * N)
    
    log_v = math.log(V) if V > 0 else 0
    log_n = math.log(N) if N > 1 else 0
    
    log_ttr = log_v / log_n if log_n > 0 and log_v > 0 else 0
    
    uber = 0.0
    if log_n > 0 and log_v > 0 and log_n != log_v:
        uber = (log_n ** 2) / (log_n - log_v)
        
    mtld = calculate_mtld(filtered_lemmas)

    # Segmented TTR helper (often referred to as Mean Segmental TTR or MSTTR)
    def get_sttr(segment_size):
        if N < segment_size:
            return ttr
        segments_ttr = []
        for i in range(0, N - segment_size + 1, segment_size):
            seg = filtered_lemmas[i:i + segment_size]
            segments_ttr.append(len(set(seg)) / segment_size)
        return sum(segments_ttr) / len(segments_ttr) if segments_ttr else ttr

    return {
        "N": N,
        "V": V,
        "TTR": round(ttr, 4),
        "RTTR": round(rttr, 4),
        "CTTR": round(cttr, 4),
        "LogTTR": round(log_ttr, 4),
        "Uber": round(uber, 4),
        "MTLD": round(mtld, 4),
        "STTR_50": round(get_sttr(50), 4),
        "STTR_100": round(get_sttr(100), 4),
        "STTR_200": round(get_sttr(200), 4)
    }

def calculate_specific_complexity(tokens, lemmas, pos_tags, pos_definitions, lang="English", wordlist_path=None):
    """
    Computes POS-dependent metrics:
    Lexical Density, Lexical Variation, Lexical Sophistication (if English wordlist matches).
    """
    # 1. Clean and filter tokens, lemmas, and tags
    valid_indices = [i for i, tok in enumerate(tokens) if not is_punctuation_or_number(tok)]
    
    N = len(valid_indices)
    if N == 0:
        return {}
        
    # Classify all tags
    pos_classes = {}
    for tag in set(pos_tags):
        if tag:
            pos_classes[tag] = classify_pos_tag(tag, pos_definitions.get(tag, ""), lang)

    # Counts
    lexical_tokens = 0
    lexical_lemmas = []
    
    verb_tokens = 0
    verb_lemmas = []
    
    noun_lemmas = []
    adj_lemmas = []
    adv_lemmas = []
    
    for idx in valid_indices:
        lemma = str(lemmas[idx]).lower()
        tag = pos_tags[idx]
        pos_class = pos_classes.get(tag, None)
        
        if pos_class:
            lexical_tokens += 1
            lexical_lemmas.append(lemma)
            
            if pos_class == "N":
                noun_lemmas.append(lemma)
            elif pos_class == "V":
                verb_tokens += 1
                verb_lemmas.append(lemma)
            elif pos_class == "Adj":
                adj_lemmas.append(lemma)
            elif pos_class == "Adv":
                adv_lemmas.append(lemma)
                
    if lexical_tokens == 0:
        return {} # specific not available
        
    # Density
    ld = lexical_tokens / N
    
    # Types
    lexical_types = len(set(lexical_lemmas))
    verb_types = len(set(verb_lemmas))
    noun_types = len(set(noun_lemmas))
    adj_types = len(set(adj_lemmas))
    adv_types = len(set(adv_lemmas))
    
    # Variation
    lv = lexical_types / lexical_tokens
    vv1 = verb_types / verb_tokens if verb_tokens > 0 else 0
    svv1 = (verb_types ** 2) / verb_tokens if verb_tokens > 0 else 0
    cvv1 = verb_types / math.sqrt(2 * verb_tokens) if verb_tokens > 0 else 0
    vv2 = verb_types / lexical_tokens
    nv = noun_types / lexical_tokens
    adjv = adj_types / lexical_tokens
    advv = adv_types / lexical_tokens
    modv = (adj_types + adv_types) / lexical_tokens
    
    # Additional NDW variations for the first 50 tokens
    filtered_lemmas = [str(lemmas[i]).lower() for i in valid_indices]
    ndw_50 = len(set(filtered_lemmas[:50])) if len(filtered_lemmas) >= 50 else len(set(filtered_lemmas))
    
    # NDW-ER50 (Expected Random 50) using math.lgamma (hypergeometric)
    ndw_er50 = 0.0
    if len(filtered_lemmas) >= 50:
        # Lemma frequency dict
        freq = {}
        for l in filtered_lemmas:
            freq[l] = freq.get(l, 0) + 1
            
        N_tot = len(filtered_lemmas)
        # Combination nCr log formula
        def log_comb(n, r):
            if r > n or r < 0: return -float('inf')
            return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
            
        denom = log_comb(N_tot, 50)
        expected_types = 0.0
        for lemma_val, count in freq.items():
            # Probability of NOT selecting this lemma at all in 50 draws
            log_prob_not = log_comb(N_tot - count, 50) - denom
            prob_not = math.exp(log_prob_not) if log_prob_not > -50 else 0.0
            expected_types += (1.0 - prob_not)
        ndw_er50 = expected_types
    else:
        ndw_er50 = len(set(filtered_lemmas))
        
    # NDW-ES50 (Expected Sequence 50) average TTR / unique lemmas of all contiguous segments of size 50
    ndw_es50 = 0.0
    if len(filtered_lemmas) >= 50:
        sums = 0
        count_seg = len(filtered_lemmas) - 49
        for i in range(count_seg):
            sums += len(set(filtered_lemmas[i:i+50]))
        ndw_es50 = sums / count_seg
    else:
        ndw_es50 = len(set(filtered_lemmas))

    results = {
        "LD": round(ld, 4),
        "LV": round(lv, 4),
        "VV1": round(vv1, 4),
        "SVV1": round(svv1, 4),
        "CVV1": round(cvv1, 4),
        "VV2": round(vv2, 4),
        "NV": round(nv, 4),
        "AdjV": round(adjv, 4),
        "AdvV": round(advv, 4),
        "ModV": round(modv, 4),
        "NDW_50": ndw_50,
        "NDW_ER50": round(ndw_er50, 4),
        "NDW_ES50": round(ndw_es50, 4),
        "_verb_tokens": verb_tokens,
        "_verb_types": verb_types,
        "_lexical_tokens": lexical_tokens,
        "_lexical_types": lexical_types,
    }
    
    # 2. Lexical Sophistication
    frequent_words = set()
    loaded_wordlist = False
    
    if wordlist_path and os.path.exists(wordlist_path):
        try:
            with open(wordlist_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts:
                        frequent_words.add(parts[-1].strip().lower())
            loaded_wordlist = True
        except Exception as e:
            print(f"Error reading custom wordlist: {e}")
            
    if not loaded_wordlist:
        # Fall back to English NGSL if language is English
        lang_clean = lang.lower() if lang else "english"
        if "english" in lang_clean or "en" in lang_clean:
            ngsl_path = os.path.join("wordlist", "english", "NGSL_1.2_wordlist.txt")
            if not os.path.exists(ngsl_path):
                ngsl_path = os.path.join("..", "wordlist", "english", "NGSL_1.2_wordlist.txt")
            if os.path.exists(ngsl_path):
                try:
                    with open(ngsl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if parts:
                                frequent_words.add(parts[-1].strip().lower())
                    loaded_wordlist = True
                except Exception as e:
                    print(f"Error reading NGSL list: {e}")
                    
    if frequent_words:
            # Sophisticated lexical tokens/types (not in frequent_words list)
            soph_lex_tokens = 0
            soph_lex_lemmas = []
            soph_verb_lemmas = []
            
            for lemma in lexical_lemmas:
                if lemma not in frequent_words:
                    soph_lex_tokens += 1
                    soph_lex_lemmas.append(lemma)
                    
            for lemma in verb_lemmas:
                if lemma not in frequent_words:
                    soph_verb_lemmas.append(lemma)
                    
            soph_lex_types = len(set(soph_lex_lemmas))
            soph_verb_types = len(set(soph_verb_lemmas))
            
            ls1 = soph_lex_tokens / lexical_tokens if lexical_tokens > 0 else 0
            ls2 = soph_lex_types / lexical_types if lexical_types > 0 else 0
            
            # Verb sophistication: use None when no verbs detected (vs 0 when verbs exist but all are common)
            if verb_tokens > 0:
                vs1 = soph_verb_types / verb_tokens
                cvs1 = soph_verb_types / math.sqrt(2 * verb_tokens)
                vs2 = (soph_verb_types ** 2) / verb_tokens
            else:
                vs1 = None
                cvs1 = None
                vs2 = None
            
            results.update({
                "LS1": round(ls1, 4),
                "LS2": round(ls2, 4),
                "VS1": round(vs1, 4) if vs1 is not None else None,
                "CVS1": round(cvs1, 4) if cvs1 is not None else None,
                "VS2": round(vs2, 4) if vs2 is not None else None,
                "_soph_verb_types": soph_verb_types,
                "_soph_verb_lemmas_sample": list(set(soph_verb_lemmas))[:10],
            })
            
    return results

def calculate_corpus_lexical_complexity(db_path, wordlist_path=None, group_by_column="filename"):
    """
    Performs full generic and specific calculations for the entire corpus
    grouped by a specific column (defaults to "filename").
    """
    if not db_path:
        return {}
        
    lang = get_corpus_language(db_path)
    pos_defs = get_pos_definitions(db_path)
    
    con = duckdb.connect(db_path)
    try:
        df = con.execute(f'SELECT "{group_by_column}", token, pos, lemma FROM corpus ORDER BY "{group_by_column}", id').fetchdf()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {}
    finally:
        con.close()
        
    if df.empty:
        return {}
        
    # Grouped results
    group_results = {}
    grouped = df.groupby(group_by_column)
    
    for group_name, group in grouped:
        tokens = group["token"].tolist()
        lemmas = group["lemma"].tolist()
        pos_tags = group["pos"].tolist()
        
        generic = calculate_generic_complexity(lemmas)
        specific = calculate_specific_complexity(tokens, lemmas, pos_tags, pos_defs, lang, wordlist_path)
        
        group_results[str(group_name)] = {
            "generic": generic,
            "specific": specific
        }
        
    # Overall corpus results
    all_tokens = df["token"].tolist()
    all_lemmas = df["lemma"].tolist()
    all_pos_tags = df["pos"].tolist()
    
    overall_generic = calculate_generic_complexity(all_lemmas)
    overall_specific = calculate_specific_complexity(all_tokens, all_lemmas, all_pos_tags, pos_defs, lang, wordlist_path)
    
    return {
        "language": lang,
        "overall": {
            "generic": overall_generic,
            "specific": overall_specific
        },
        "files": group_results
    }
