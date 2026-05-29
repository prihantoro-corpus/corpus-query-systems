import streamlit as st
import pandas as pd
import time
import logging
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

_logger = logging.getLogger("cortex_profiler")

# ---------------------------------------------------------------------------
# NOTE on decorator order:
#   @profile_func and @notify_timing MUST wrap the *cached* function so they
#   run on EVERY call (including cache hits, if desired) or at least don't
#   interfere with cache-key hashing.
#
#   @st.cache_data MUST be the INNERMOST decorator (directly above `def`)
#   so it sees the real function signature and can hash its arguments.
#   Wrapping st.cache_data with notify_timing/profile_func caused those
#   decorators to become part of the cached object, breaking hashing of
#   any mutable kwargs (especially xml_params as a list).
# ---------------------------------------------------------------------------


# --- Internal helper: run a function with timing logged ---
def _timed(label, func, *args, **kwargs):
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    _logger.info(f"[CACHED] {label} took {time.perf_counter() - t0:.4f}s")
    return result


# -------------------------------------------------------------------------
# Dictionary / Lemma
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching Dictionary...")
def _cached_get_lemma_details(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    return get_all_lemma_forms_details(
        db_path, word,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params),
        **kwargs
    )

def cached_get_lemma_details(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("get_lemma_details", _cached_get_lemma_details,
                  db_path, word,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params, **kwargs)


@st.cache_data(show_spinner="Extracting Contexts...")
def _cached_get_context_ngrams(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    return get_detailed_contextual_ngrams(
        db_path, word,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params),
        **kwargs
    )

def cached_get_context_ngrams(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("get_context_ngrams", _cached_get_context_ngrams,
                  db_path, word,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params, **kwargs)


@st.cache_data(show_spinner="Formatting Examples...")
def _cached_get_dict_examples(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    return get_dictionary_examples(
        db_path, word,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params),
        **kwargs
    )

def cached_get_dict_examples(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("get_dict_examples", _cached_get_dict_examples,
                  db_path, word,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params, **kwargs)


@st.cache_data(show_spinner="Fetching Random Examples...")
def _cached_get_random_examples(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    return get_random_examples(
        db_path, word,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params),
        **kwargs
    )

def cached_get_random_examples(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("get_random_examples", _cached_get_random_examples,
                  db_path, word,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params, **kwargs)


# -------------------------------------------------------------------------
# Sub-corpus size
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Calculating sub-corpus size...")
def _cached_get_subcorpus_size(db_path, xml_where_clause="", xml_params=()):
    return get_subcorpus_size(
        db_path,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params)
    )

def cached_get_subcorpus_size(db_path, xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("get_subcorpus_size", _cached_get_subcorpus_size,
                  db_path,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params)


# -------------------------------------------------------------------------
# Related forms
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Searching related forms...")
def _cached_get_related_forms(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    return get_related_forms_by_regex(
        db_path, word,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params),
        **kwargs
    )

def cached_get_related_forms(db_path, word, xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("get_related_forms", _cached_get_related_forms,
                  db_path, word,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params, **kwargs)


# -------------------------------------------------------------------------
# KWIC / Concordance
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating KWIC...")
def _cached_generate_kwic(db_path, query, left, right, corpus_name,
                           xml_where_clause="", xml_params=(), **kwargs):
    """
    xml_params must be a tuple here so st.cache_data can hash it.
    We convert it back to list before passing to the engine.
    """
    return generate_kwic(
        db_path, query, left, right, corpus_name,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params),
        **kwargs
    )

def cached_generate_kwic(db_path, query, left, right, corpus_name,
                          xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("generate_kwic", _cached_generate_kwic,
                  db_path, query, left, right, corpus_name,
                  xml_where_clause=xml_where_clause,
                  xml_params=xml_params, **kwargs)


# -------------------------------------------------------------------------
# Collocations
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating Collocations...")
def _cached_generate_collocation(db_path, word, window, min_freq, max_rows, is_raw,
                                  corpus_stats, token_filter="", pos_filter="",
                                  lemma_filter="", stat_measure="log-likelihood",
                                  xml_where_clause="", xml_params=()):
    return generate_collocation_results(
        db_path, word, window, min_freq, max_rows, is_raw,
        token_filter=token_filter,
        pos_filter=pos_filter,
        lemma_filter=lemma_filter,
        corpus_stats=corpus_stats,
        stat_measure=stat_measure,
        xml_where_clause=xml_where_clause,
        xml_params=list(xml_params)
    )

def cached_generate_collocation(db_path, word, window, min_freq, max_rows, is_raw,
                                 corpus_stats, token_filter="", pos_filter="",
                                 lemma_filter="", stat_measure="log-likelihood",
                                 xml_where_clause="", xml_params=(), **kwargs):
    xml_params = tuple(xml_params) if xml_params else ()
    return _timed("generate_collocation", _cached_generate_collocation,
                  db_path, word, window, min_freq, max_rows, is_raw,
                  corpus_stats, token_filter, pos_filter, lemma_filter,
                  stat_measure, xml_where_clause, xml_params)


# -------------------------------------------------------------------------
# N-Grams
# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating N-Grams...")
def _cached_generate_ngrams(db_path, n, filters, is_raw, corpus_name, **kwargs):
    return generate_n_grams_v2(db_path, n, filters, is_raw, corpus_name, **kwargs)

def cached_generate_ngrams(db_path, n, filters, is_raw, corpus_name, **kwargs):
    return _timed("generate_ngrams", _cached_generate_ngrams,
                  db_path, n, filters, is_raw, corpus_name, **kwargs)
