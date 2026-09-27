================================================================================
CORTEX: Advanced Corpus Linguistics & Text Analytics Engine
TECHNOLOGICAL ARCHITECTURE OVERVIEW
================================================================================

1. OVERVIEW & PURPOSE
--------------------------------------------------------------------------------
Official Website: https://www.cortex-app.org/

Cortex is a comprehensive, high-performance corpus linguistics and text analytics platform.
It combines ultra-fast SQL-based corpus querying with state-of-the-art Natural Language 
Processing (NLP) pipelines, statistical association metrics, machine learning, and interactive 
data visualization tools.

2. ARCHITECTURE LAYERS & DESIGN PATTERNS
--------------------------------------------------------------------------------
The architecture follows a decoupled, modular 3-tier architecture:

[ USER INTERFACE LAYER ]
    └── Streamlit Web Application (ui_streamlit/cortex.py & ui_streamlit/views/)
    └── Interactive UI Components (PyVis network graphs, Plotly, Altair, Wordclouds)

[ CORE ENGINE & PROCESSING LAYER ]
    └── Core Orchestrator (core/config.py, core/ai_service.py)
    └── Processing Modules (core/modules/: concordance, collocation, ngram, stats, etc.)
    └── NLP Pipeline Adapters (Stanza, spaCy, NLTK, Transformers, BERTopic)

[ DATA & STORAGE LAYER ]
    └── DuckDB Analytics Database (Embedded vectorized columnar SQL engine)
    └── XML/Metadata Corpus Storage (TEI/XML compliant structured corpora)
    └── SQLite Intermediates & Caching Engine (ui_streamlit/caching.py)

3. CORE COMPONENTS & MODULE BREAKDOWN
--------------------------------------------------------------------------------
• UI Layer (Streamlit App & Views):
  - `ui_streamlit/cortex.py`: Entry point and navigation router.
  - `ui_streamlit/views/`: Dedicated view modules for Concordance, Collocation, N-Grams, 
    Keyword Analysis, Lexical Complexity, Readability, Word Trends, and LLM Analysis.

• Corpus Storage & Database Engine (DuckDB):
  - Embedded columnar relational database tuned for ultra-fast text querying, token matching, 
    vectorized aggregations, and sub-second KWIC (Key Word In Context) extraction.
  - Native support for XML corpus parsing, structured metadata filtering (speaker, genre, year), 
    and fast multi-token n-gram matrix operations.

• Natural Language Processing (NLP) & Tokenization Pipeline:
  - Multi-engine NLP support including spaCy, Stanza, NLTK, TreeTagger, Jieba (Chinese), 
    and MeCab/UniDic (Japanese).
  - Handles POS (Part-of-Speech) tagging, Lemmatization, Dependency Parsing, Named Entity 
    Recognition (NER), and Grapheme-to-Phoneme (G2P) conversion.

• Statistical & Analytical Computing Engine:
  - Measures of Association: Mutual Information (MI), MI3, Log-Likelihood (LL), t-score, 
    z-score, and Dice coefficient for collocation analysis.
  - Dispersion & Keyword Analysis: Log-Likelihood ratio, Chi-Square, Relative Frequency, 
    and Juilland's D dispersion metrics.
  - Text Complexity & Readability: Automated Readability Index (ARI), Flesch-Kincaid, 
    Coleman-Liau, Gunning Fog, and CEFR level estimations.

• AI & Machine Learning Integrations:
  - BERTopic & Sentence Transformers for neural topic modeling and semantic clustering.
  - Hugging Face Transformers & LLM API integrations (`core/ai_service.py`) for automated 
    text summarization, exercise generation, and pedagogical quiz creation.

4. TECHNOLOGY STACK SUMMARY
--------------------------------------------------------------------------------
• Programming Language: Python 3.10+
• Web Framework: Streamlit
• Analytics Database: DuckDB
• Data Processing: Pandas, NumPy, SciPy, Scikit-learn, LXML
• NLP Frameworks: spaCy, Stanza, NLTK, Transformers, BERTopic, g2p-en, eng-to-ipa
• Visualization: Plotly, Altair, PyVis (graph networks), Matplotlib, WordCloud
• Deployment Target: Local & Hugging Face Spaces (Docker / Streamlit Engine)

================================================================================
