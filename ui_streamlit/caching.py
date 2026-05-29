import streamlit as st
import pandas as pd
from core.modules.dictionary_service import (
    get_all_lemma_forms_details, 
    get_detailed_contextual_ngrams, 
    get_dictionary_examples,
    get_random_examples,
    get_related_forms_by_regex,
    get_subcorpus_size
)
from core.modules.concordance import generate_kwic
from core.modules.ngram import generate_n_grams_v2
from core.modules.collocation import generate_collocation_results
from core.utils.profiler import profile_func
from ui_streamlit.utils import notify_timing

@st.cache_data(show_spinner="Searching Dictionary...")
@notify_timing("Dictionary search completed")
@profile_func
def cached_get_lemma_details(db_path, word, **kwargs):
    return get_all_lemma_forms_details(db_path, word, **kwargs)

@st.cache_data(show_spinner="Extracting Contexts...")
@notify_timing("Context extraction completed")
@profile_func
def cached_get_context_ngrams(db_path, word, **kwargs):
    return get_detailed_contextual_ngrams(db_path, word, **kwargs)

@st.cache_data(show_spinner="Formatting Examples...")
@notify_timing("Examples formatted")
@profile_func
def cached_get_dict_examples(db_path, word, **kwargs):
    return get_dictionary_examples(db_path, word, **kwargs)

@st.cache_data(show_spinner="Fetching Random Examples...")
@notify_timing("Random examples fetched")
@profile_func
def cached_get_random_examples(db_path, word, **kwargs):
    return get_random_examples(db_path, word, **kwargs)

@st.cache_data(show_spinner="Calculating sub-corpus size...")
@notify_timing("Sub-corpus size calculated")
@profile_func
def cached_get_subcorpus_size(db_path, **kwargs):
    return get_subcorpus_size(db_path, **kwargs)

@st.cache_data(show_spinner="Searching related forms...")
@notify_timing("Related forms search completed")
@profile_func
def cached_get_related_forms(db_path, word, **kwargs):
    return get_related_forms_by_regex(db_path, word, **kwargs)

@st.cache_data(show_spinner="Generating KWIC...")
@notify_timing("Concordance generated")
@profile_func
def cached_generate_kwic(db_path, query, left, right, corpus_name, **kwargs):
    return generate_kwic(db_path, query, left, right, corpus_name, **kwargs)

@st.cache_data(show_spinner="Generating Collocations...")
@notify_timing("Collocations generated")
@profile_func
def cached_generate_collocation(db_path, word, window, min_freq, max_rows, is_raw, corpus_stats, 
                               token_filter="", pos_filter="", lemma_filter="", stat_measure="log-likelihood", **kwargs):
    return generate_collocation_results(
        db_path, word, window, min_freq, max_rows, is_raw, 
        token_filter=token_filter, 
        pos_filter=pos_filter, 
        lemma_filter=lemma_filter, 
        corpus_stats=corpus_stats, 
        stat_measure=stat_measure,
        **kwargs
    )

@st.cache_data(show_spinner="Generating N-Grams...")
@notify_timing("N-Grams generated")
@profile_func
def cached_generate_ngrams(db_path, n, filters, is_raw, corpus_name, **kwargs):
    return generate_n_grams_v2(db_path, n, filters, is_raw, corpus_name, **kwargs)
