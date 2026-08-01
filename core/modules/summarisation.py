import duckdb
import pandas as pd
import re
import math
from collections import Counter

# Try importing sumy for extractive summarization
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.kl import KLSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False

def get_summarization_metadata_fields(db_path):
    """
    Returns a list of metadata columns available in the corpus table.
    """
    if not db_path:
        return []
    
    con = duckdb.connect(db_path)
    try:
        standard_cols = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low'}
        all_cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        meta_cols = [c for c in all_cols if c not in standard_cols]
        # Exclude internal columns
        meta_cols = [c for c in meta_cols if not (c.endswith('_len') or c.endswith('_start') or c.endswith('_id'))]
        return meta_cols
    except Exception as e:
        print(f"Error getting metadata fields: {e}")
        return []
    finally:
        con.close()

def get_metadata_values(db_path, field):
    """
    Returns unique values for a specific metadata field.
    """
    if not db_path or not field:
        return []
    
    con = duckdb.connect(db_path)
    try:
        query = f"SELECT DISTINCT {field} FROM corpus WHERE {field} IS NOT NULL ORDER BY {field}"
        df = con.execute(query).fetch_df()
        return df[field].tolist()
    except Exception as e:
        print(f"Error getting metadata values: {e}")
        return []
    finally:
        con.close()

def get_text_data(db_path, basis="Overall", field=None, value=None):
    """
    Fetches text from the corpus based on the selected basis.
    """
    if not db_path:
        return ""
    
    con = duckdb.connect(db_path)
    try:
        where_clause = ""
        params = []
        if basis == "By Metadata" and field and value:
            where_clause = f"WHERE {field} = ?"
            params = [value]
        elif basis == "By File Name" and field and value: # Usually field name is 'filename' or similar
            where_clause = f"WHERE {field} = ?"
            params = [value]
            
        # We join tokens by spaces, but try to preserve sentence boundaries if possible
        # DuckDB can aggregate tokens.
        query = f"SELECT token FROM corpus {where_clause} ORDER BY id"
        df = con.execute(query, params).fetch_df()
        
        if df.empty:
            return ""
        
        # Simple join. For better results, we might want to handle punctuation-based spacing
        return " ".join(df['token'].astype(str).tolist())
    except Exception as e:
        print(f"Error fetching text: {e}")
        return ""
    finally:
        con.close()

def summarize_text_extractive(text, language="english", word_limit=100, algorithm="Luhn"):
    """
    Summarizes text using Sumy if available, otherwise falls back to a simple word-frequency based extractive method.
    """
    if not text or len(text.split()) < word_limit:
        return text

    if SUMY_AVAILABLE:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer(language))
            stemmer = Stemmer(language)
            
            if algorithm == "Lsa":
                summarizer = LsaSummarizer(stemmer)
            elif algorithm == "LexRank":
                summarizer = LexRankSummarizer(stemmer)
            elif algorithm == "KL":
                summarizer = KLSummarizer(stemmer)
            else:
                summarizer = LuhnSummarizer(stemmer)
                
            summarizer.stop_words = get_stop_words(language)
            
            # Sumy works by sentence count, not word count. 
            # We estimate sentences based on average sentence length (~20 words)
            sentence_count = max(1, math.ceil(word_limit / 20))
            
            summary_sentences = summarizer(parser.document, sentence_count)
            return " ".join([str(s) for s in summary_sentences])
        except Exception as e:
            print(f"Sumy error: {e}. Falling back to simple summarizer.")
            
    # Fallback: Simple frequency based extractive summarization
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        return text
    
    # Tokenize and count frequencies (simple)
    words = re.findall(r'\b\w+\b', text.lower())
    freqs = Counter(words)
    
    # Score sentences
    scores = []
    for sent in sentences:
        sent_words = re.findall(r'\b\w+\b', sent.lower())
        if not sent_words:
            scores.append(0)
            continue
        score = sum(freqs.get(w, 0) for w in sent_words) / len(sent_words)
        scores.append(score)
    
    # Pick top sentences
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    summary_sentences = []
    current_word_count = 0
    # Keep original order
    selected_indices = []
    
    for idx, score in indexed_scores:
        sent = sentences[idx]
        sent_len = len(sent.split())
        if current_word_count + sent_len <= word_limit * 1.2: # Allow slight overflow
            selected_indices.append(idx)
            current_word_count += sent_len
        if current_word_count >= word_limit:
            break
            
    selected_indices.sort()
    return " ".join([sentences[i] for i in selected_indices])

def summarize_text_ai(text, provider="Ollama", model="llama3", api_key=None, word_limit=100):
    """
    Summarizes text using an AI provider (Ollama or Gemini).
    """
    from core.ai_service import get_ai_response
    
    prompt = f"Summarize the following text in approximately {word_limit} words. Focus on the main topics and themes:\n\n{text[:10000]}" # Limit input to 10k chars for safety
    
    response, error = get_ai_response(prompt, provider, model, api_key)
    return response if response else f"AI Error: {error}"
