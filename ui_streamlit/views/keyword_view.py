import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
import importlib
import core.modules.keyword
from ui_streamlit.utils import notify_timing
importlib.reload(core.modules.keyword)
from core.modules.keyword import generate_keyword_list, generate_grouped_keyword_list
from core.visualiser.wordcloud import generate_wordcloud
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.io_utils import df_to_excel_bytes, dfs_to_zip_excel_bytes
from core.config import get_available_corpora, BUILT_IN_CORPORA
from core.preprocessing.corpus_loader import load_monolingual_corpus_files, load_built_in_corpus
from core.preprocessing.xml_parser import apply_xml_restrictions, get_xml_attribute_columns
from core.ai_service import parse_nl_query
import io
import duckdb

def parse_frequency_list_file(uploaded_file):
    """Parses a word-tab-freq or word-space-freq file."""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.strip().split('\n')
        data = []
        total = 0
        for line in lines:
            parts = line.strip().split() # Handles both tab and spaces
            if len(parts) >= 2:
                word = parts[0].lower()
                try:
                    freq = int(parts[1])
                    data.append({'token': word, 'freq': freq})
                    total += freq
                except ValueError:
                    continue
        return pd.DataFrame(data), total
    except Exception as e:
        st.error(f"Error parsing frequency list: {e}")
        return None, 0

def render_keyword_view():
    st.header("Keyword Analysis")

    with st.expander("💡 **Method & Transparency: Keyword Analysis**", expanded=False):
        st.markdown("""
        **Goal:** Identify words that are unusually frequent (or infrequent) in your **Target Corpus** compared to a **Reference Corpus**.
        
        **Data Used:** 
        - Word frequencies in the **Target Corpus** (or search results).
        - Word frequencies in a **Reference Corpus** (e.g., Brown Corpus, BNC, or another uploaded file).
        
        **Statistical Measures:**
        - **Log-Likelihood (LL):** Measures **statistical significance** (evidence strength). High LL means the difference in usage is very unlikely to be due to chance.
        - **Log Ratio:** Measures **effect size** (magnitude of difference). A Log Ratio of +3 means the word is $2^3 = 8$ times more frequent in the target than in the reference.
        
        **Keyword Types:**
        - **Positive Keywords (High Keyness):** Words that are used significantly **more** in the target than in the reference.
        - **Negative Keywords (Low Keyness):** Words that are used significantly **less** (or are missing) in the target compared to the reference.
        """)
    
    # 1. Corpus Selection
    current_path = get_state('current_corpus_path')
    current_name = get_state('current_corpus_name', 'Corpus')
    
    if not current_path:
        st.warning("Please load a primary corpus first.")
        return

    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Keyword")

    with col_main:


        # 1. Comparison Mode & Reference Selection
        comp_mode = get_state('comparison_mode', False)
        ref_path = get_state('comp_corpus_path')
        ref_name = get_state('comp_corpus_name')

        with st.container(border=True):
            st.markdown("#### 🎯 Reference Corpus Selection")

            if comp_mode and ref_path:
                st.success(f"**Using Global Comparison Corpus:** {ref_name}")
                st.caption("You can change the comparison corpus in the Sidebar.")

            elif not ref_path:
                st.info("Select a reference corpus to compare against.")
                tabs = st.tabs(["🏛️ Pre-built", "📤 Upload"])

                with tabs[0]:
                    available_corpora = get_available_corpora()
                    sel_name = st.radio("Built-in Corpora", list(available_corpora.keys()), horizontal=True)
                    if st.button("Load as Reference", key="load_builtin_ref"):
                        with st.spinner("Downloading and processing..."):
                            result = load_built_in_corpus(sel_name, available_corpora[sel_name])
                            if result.get('error'):
                                st.error(result['error'])
                            else:
                                set_state('comp_corpus_path', result['db_path'])
                                set_state('comp_corpus_stats', result['stats'])
                                set_state('comp_corpus_name', sel_name)
                                set_state('comp_xml_structure_data', result.get('structure'))
                                st.success(f"Loaded {sel_name} as reference.")
                                st.rerun()

                with tabs[1]:
                    uploaded_ref = st.file_uploader("Upload XML or Frequency List", type=['xml', 'txt', 'csv', 'tsv'], key="upload_ref_kw")
                    if uploaded_ref:
                        if st.button("Process Reference", key="btn_process_ref"):
                            with st.spinner("Processing reference..."):
                                if uploaded_ref.name.endswith('.xml'):
                                    result = load_monolingual_corpus_files([uploaded_ref], explicit_lang_code=get_state('target_lang', 'en'), selected_format='.xml / auto')
                                    if result.get('error'):
                                        st.error(result['error'])
                                    else:
                                        set_state('comp_corpus_path', result['db_path'])
                                        set_state('comp_corpus_stats', result['stats'])
                                        set_state('comp_corpus_name', uploaded_ref.name)
                                        set_state('comp_ref_type', 'db')
                                        st.success("Reference corpus loaded.")
                                        st.rerun()
                                else:
                                    df_freq, total = parse_frequency_list_file(uploaded_ref)
                                    if df_freq is not None and not df_freq.empty:
                                        set_state('comp_freq_df', df_freq)
                                        set_state('comp_total_tokens', total)
                                        set_state('comp_corpus_name', uploaded_ref.name)
                                        set_state('comp_corpus_path', 'frequency_list') 
                                        set_state('comp_ref_type', 'freq_list')
                                        st.success(f"Loaded {len(df_freq)} entries.")
                                        st.rerun()
                return 
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.success(f"**Reference:** {ref_name} (Manual/Previous)")
                with c2:
                    if st.button("Switch Reference", key="btn_switch_ref"):
                        set_state('comp_corpus_path', None)
                        set_state('comp_corpus_name', None)
                        set_state('comp_freq_df', None)
                        set_state('comp_total_tokens', 0)
                        st.rerun()

        st.markdown("---")

        if not comp_mode:
            p = _render_kw_controls(current_path, current_name, ref_path, ref_name)
            if p:
                _run_keyword_analysis('primary', current_path, current_name, p, get_state)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Target: {current_name}")
                p1 = _render_kw_controls(current_path, current_name, ref_path, ref_name, "c1")
                if p1:
                    _run_keyword_analysis('primary', current_path, current_name, p1, get_state)
            with col2:
                st.subheader(f"Target: {ref_name}")
                p2 = _render_kw_controls(ref_path, ref_name, current_path, current_name, "c2")
                if p2:
                    _run_keyword_analysis('secondary', ref_path, ref_name, p2, get_state)

        st.markdown("---")
        if not comp_mode:
            res = st.session_state.get('last_kw_results_primary')
            if res:
                render_keyword_results(res)
        else:
            c1, c2 = st.columns(2)
            with c1:
                res1 = st.session_state.get('last_kw_results_primary')
                if res1:
                    render_keyword_results(res1, "c1")
            with c2:
                res2 = st.session_state.get('last_kw_results_secondary')
                if res2:
                    render_keyword_results(res2, "c2")


def _render_kw_controls(target_path, target_name, default_ref_path, default_ref_name, key_suffix=""):
    suffix = f"_{key_suffix}" if key_suffix else ""
    params = {}
    st.markdown("##### Settings")

    ref_options = []
    if default_ref_path:
        ref_options.append(f"Corpus: {default_ref_name}")

    available = get_available_corpora()
    ref_options.extend([f"Built-in: {k}" for k in available.keys()])
    sel_ref_label = st.radio("Reference Corpus", ref_options, index=0, horizontal=True, key=f"kw_ref_sel{suffix}")

    ref_path_selected = None
    if sel_ref_label.startswith("Corpus: "):
        ref_path_selected = default_ref_path
        ref_name_selected = default_ref_name
    else:
        b_name = sel_ref_label.replace("Built-in: ", "")
        ref_path_selected = f"BUILTIN:{b_name}"
        ref_name_selected = b_name

    c1, c2 = st.columns(2)
    with c1:
        min_freq = st.number_input("Min Freq (Target)", 1, 1000, 5, key=f"kw_min{suffix}")
    with c2:
        top_n = st.number_input("Top N", 10, 500, 50, key=f"kw_top{suffix}")

    p_val = st.radio("P-Value Cutoff", ["0.05", "0.01", "0.001", "None"], index=2, horizontal=True, key=f"kw_pval{suffix}")

    st.markdown("##### Analysis Basis (Optional)")
    by_file_opt = st.checkbox("By Individual File", value=False, key=f"kw_by_file{suffix}", help="Generate separate keyword lists for each file in the target corpus.")
    by_subcorpus_opt = st.checkbox("By Sub-corpora Attributes", value=False, key=f"kw_by_sub{suffix}", help="Generate separate keyword lists for each XML attribute value (sub-corpora).")

    st.markdown("##### Target Restrictions")
    view_name = f"kw_target_{key_suffix}" if key_suffix else "kw_target"
    xml_filters = render_xml_restriction_filters(target_path, view_name, corpus_name=target_name)
    xw, xp = apply_xml_restrictions(xml_filters)

    if st.button("Calculate Keywords", type="primary", key=f"btn_kw{suffix}"):
        params['min_freq'] = min_freq
        params['top_n'] = top_n
        params['p_val'] = p_val
        params['ref_path'] = ref_path_selected
        params['ref_name'] = ref_name_selected
        params['xml_where'] = xw
        params['xml_params'] = xp
        params['by_file'] = by_file_opt
        params['by_subcorpus'] = by_subcorpus_opt
        return params
    return None


def _run_keyword_analysis(identifier, target_path, target_name, params, state_getter):
    ref_path = params['ref_path']
    ref_name = params['ref_name']
    final_ref_path = ref_path

    if ref_path and ref_path.startswith("BUILTIN:"):
        b_name = ref_path.replace("BUILTIN:", "")
        with st.spinner(f"Loading '{b_name}'..."):
            res = load_built_in_corpus(b_name, BUILT_IN_CORPORA[b_name])
            if not res.get('error'):
                final_ref_path = res['db_path']
            else:
                st.error(res['error'])
                return

    with st.spinner(f"Calculating Keywords: {target_name} vs {ref_name}..."):
        def apply_p_val_filter(df, p_cutoff):
            if df is None or df.empty or p_cutoff == "None":
                return df
            is_stable = df['Type'] == 'Stable'
            if p_cutoff == "0.001":
                is_sig = df['Significance'].str.contains(r'\*\*\*', na=False)
            elif p_cutoff == "0.01":
                is_sig = df['Significance'].str.contains(r'\*\*', na=False)
            elif p_cutoff == "0.05":
                is_sig = df['Significance'] != 'ns'
            else:
                is_sig = pd.Series(True, index=df.index)
            return df[is_stable | is_sig]

        # Custom handling for frequency list reference
        ref_freq_df = None
        ref_total_tokens = 0
        if ref_path == 'frequency_list':
            ref_freq_df = st.session_state.get('comp_freq_df')
            ref_total_tokens = st.session_state.get('comp_total_tokens', 0)
            final_ref_path = None # Ensure it doesn't try to connect to a DB

        df_overall = notify_timing("Keyword analysis completed")(generate_keyword_list)(
            target_path, 
            ref_db_path=final_ref_path, 
            target_xml_where=params['xml_where'], 
            target_xml_params=params['xml_params'], 
            ref_freq_df=ref_freq_df,
            ref_total_tokens=ref_total_tokens,
            min_freq=params['min_freq']
        )
        df_overall = apply_p_val_filter(df_overall, params['p_val'])

        by_filename = {}
        if params.get('by_file', False):
            by_filename = generate_grouped_keyword_list(
                target_path, 
                group_by_col="filename", 
                ref_db_path=final_ref_path, 
                target_xml_where=params['xml_where'], 
                target_xml_params=params['xml_params'], 
                ref_freq_df=ref_freq_df,
                ref_total_tokens=ref_total_tokens,
                min_freq=params['min_freq']
            )
            for k in by_filename:
                by_filename[k] = apply_p_val_filter(by_filename[k], params['p_val'])

        by_attributes = {}
        if params.get('by_subcorpus', False):
            con_temp = duckdb.connect(target_path, read_only=True)
            attr_cols = get_xml_attribute_columns(con_temp)
            # Filter attributes to only keep those with <= 100 unique values (to avoid doc_id, titles, details, etc.)
            filtered_attr_cols = []
            if attr_cols:
                for attr in attr_cols:
                    if attr == "filename":
                        continue
                    try:
                        num_unique = con_temp.execute(f"SELECT count(DISTINCT \"{attr}\") FROM corpus").fetchone()[0]
                        if num_unique <= 100:
                            filtered_attr_cols.append(attr)
                        else:
                            st.info(f"Skipping attribute '{attr}' for sub-corpora logic: has too many unique values ({num_unique}).")
                    except:
                        # Fallback: keep it if query fails
                        filtered_attr_cols.append(attr)
            con_temp.close()

            if filtered_attr_cols:
                for attr in filtered_attr_cols:
                    grouped = generate_grouped_keyword_list(
                        target_path, 
                        attr, 
                        ref_db_path=final_ref_path, 
                        target_xml_where=params['xml_where'], 
                        target_xml_params=params['xml_params'], 
                        ref_freq_df=ref_freq_df,
                        ref_total_tokens=ref_total_tokens,
                        min_freq=params['min_freq']
                    )
                    if grouped:
                        for k in grouped:
                            grouped[k] = apply_p_val_filter(grouped[k], params['p_val'])
                        by_attributes[attr] = grouped

        st.session_state[f'last_kw_results_{identifier}'] = {
            'overall': df_overall, 'by_filename': by_filename, 'by_attributes': by_attributes,
            'target': target_name, 'ref': ref_name, 'top_n': params['top_n']
        }


def render_keyword_results(res, key_suffix=""):
    top_n = res['top_n']
    st.markdown(f"### Results for: **{res['target']}**")
    st.caption(f"Compared against: {res['ref']}")

    def _get_top_list(df, n=10, category='Positive'):
        if df is None or df.empty:
            return ""
        filtered = df[df['Type'] == category]
        if category == 'Negative':
            filtered = filtered.sort_values('LL', ascending=False)
        elif category == 'Stable':
            filtered = filtered.sort_values('LL', ascending=True)
        return ", ".join(filtered.head(n)['token'].tolist())

    o_rows_pos, o_rows_neg, o_rows_stable = [], [], []
    o_rows_pos.append({"Classification": "Overall", "Keywords": _get_top_list(res.get('overall'), category='Positive')})
    o_rows_neg.append({"Classification": "Overall", "Keywords": _get_top_list(res.get('overall'), category='Negative')})
    o_rows_stable.append({"Classification": "Overall", "Keywords": _get_top_list(res.get('overall'), category='Stable')})

    for fname, df in res.get('by_filename', {}).items():
        o_rows_pos.append({"Classification": f"File: {fname}", "Keywords": _get_top_list(df, category='Positive')})
        o_rows_neg.append({"Classification": f"File: {fname}", "Keywords": _get_top_list(df, category='Negative')})
        o_rows_stable.append({"Classification": f"File: {fname}", "Keywords": _get_top_list(df, category='Stable')})
    for attr, groups in res.get('by_attributes', {}).items():
        for val, df in groups.items():
            o_rows_pos.append({"Classification": f"{attr}={val}", "Keywords": _get_top_list(df, category='Positive')})
            o_rows_neg.append({"Classification": f"{attr}={val}", "Keywords": _get_top_list(df, category='Negative')})
            o_rows_stable.append({"Classification": f"{attr}={val}", "Keywords": _get_top_list(df, category='Stable')})

    df_o_pos, df_o_neg, df_o_stable = pd.DataFrame(o_rows_pos), pd.DataFrame(o_rows_neg), pd.DataFrame(o_rows_stable)

    st.markdown("#### 📊 Keyword Overview (Top 10)")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown("**High Keyness (Positive)** <span style='cursor: help;' title='Positive keywords are words used significantly MORE in the target corpus compared to the reference corpus (Log-Likelihood >= 3.84, LogRatio > 0). The threshold 3.84 corresponds to p < 0.05.'>❓</span>", unsafe_allow_html=True)
        st.dataframe(df_o_pos, use_container_width=True, hide_index=True)
    with oc2:
        st.markdown("**Low Keyness (Negative)** <span style='cursor: help;' title='Negative keywords are words used significantly LESS in the target corpus compared to the reference corpus (Log-Likelihood >= 3.84, LogRatio < 0). The threshold 3.84 corresponds to p < 0.05.'>❓</span>", unsafe_allow_html=True)
        st.dataframe(df_o_neg, use_container_width=True, hide_index=True)
    with oc3:
        st.markdown("**Comparable (Stable)** <span style='cursor: help;' title='Stable words are those that occur with comparable frequencies in both corpora and fail to reach a statistically significant difference (Log-Likelihood < 3.84, which corresponds to p >= 0.05).'>❓</span>", unsafe_allow_html=True)
        st.dataframe(df_o_stable, use_container_width=True, hide_index=True)

    all_dfs = {
        'Overview_Positive': df_o_pos, 
        'Overview_Negative': df_o_neg,
        'Overview_Stable': df_o_stable
    }
    if res.get('overall') is not None:
        all_dfs['Overall'] = res['overall']
    for fname, df in res.get('by_filename', {}).items():
        safe_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in fname])
        all_dfs[f"File_{safe_name}"] = df
    for attr, groups in res.get('by_attributes', {}).items():
        for val, df in groups.items():
            safe_val = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in str(val)])
            all_dfs[f"Attr_{attr}_{safe_val}"] = df

    if all_dfs:
        zip_data = dfs_to_zip_excel_bytes(all_dfs)
        st.download_button(label="📥 Download All Keywords (ZIP)", data=zip_data, file_name=f"keywords_{res['target'].replace(' ', '_')}.zip", mime="application/zip", key=f"dl_kw_zip_{key_suffix}")

    st.markdown("---")
    st.markdown("#### 🔍 Detailed Analysis")

    def _draw_kw_table(df, title_prefix, sub_key):
        if df is None or df.empty:
            st.info(f"No keywords for {title_prefix}.")
            return
        pos = df[df['Type'] == 'Positive'].head(top_n)
        neg = df[df['Type'] == 'Negative'].sort_values('LL', ascending=False).head(top_n)
        stable = df[df['Type'] == 'Stable'].sort_values('LL', ascending=True).head(top_n)
        tab_p, tab_n, tab_s = st.tabs([
            f"High Keyness ({len(pos)})", 
            f"Low Keyness ({len(neg)})", 
            f"Stable ({len(stable)})"
        ])
        with tab_p:
            st.caption("ℹ️ **High Keyness (Positive)**: Words significantly overused in target (Log-Likelihood ≥ 3.84, LogRatio > 0, corresponding to $p < 0.05$).")
            if not pos.empty:
                fd = dict(zip(pos['token'], pos['LL']))
                fig = generate_wordcloud(fd, title=f"Positive: {title_prefix}", width=400, height=200)
                if fig:
                    st.pyplot(fig)
                st.dataframe(pos[['token', 'LL', 'LogRatio', 'Significance']], use_container_width=True, height=250, hide_index=True)
        with tab_n:
            st.caption("ℹ️ **Low Keyness (Negative)**: Words significantly underused in target (Log-Likelihood ≥ 3.84, LogRatio < 0, corresponding to $p < 0.05$).")
            if not neg.empty:
                fd = dict(zip(neg['token'], neg['LL']))
                fig = generate_wordcloud(fd, title=f"Negative: {title_prefix}", width=400, height=200)
                if fig:
                    st.pyplot(fig)
                st.dataframe(neg[['token', 'LL', 'LogRatio', 'Significance']], use_container_width=True, height=250, hide_index=True)
        with tab_s:
            st.caption("ℹ️ **Comparable (Stable)**: Words with similar frequencies that fail to reach statistical significance (Log-Likelihood < 3.84, corresponding to $p \ge 0.05$).")
            if not stable.empty:
                fd = dict(zip(stable['token'], stable['freq_t']))
                fig = generate_wordcloud(fd, title=f"Stable: {title_prefix}", width=400, height=200)
                if fig:
                    st.pyplot(fig)
                st.dataframe(stable[['token', 'freq_t', 'freq_r', 'LL', 'LogRatio', 'Significance']], use_container_width=True, height=250, hide_index=True)

    with st.expander("🌍 Overall Corpus Keywords", expanded=True):
        _draw_kw_table(res.get('overall'), "Overall", f"ov_{key_suffix}")
    
    by_file = res.get('by_filename', {})
    if by_file:
        with st.expander("📁 Keywords By Filename", expanded=False):
            file_list = list(by_file.keys())
            selected_file = st.selectbox("Select File", file_list, key=f"sb_file_{key_suffix}")
            if selected_file:
                _draw_kw_table(by_file[selected_file], selected_file, f"f_{selected_file}_{key_suffix}")
                    
    by_attr = res.get('by_attributes', {})
    if by_attr:
        for attr_name, groups in by_attr.items():
            with st.expander(f"🏷️ Keywords By {attr_name.title()}", expanded=False):
                val_list = list(groups.keys())
                selected_val = st.selectbox(f"Select {attr_name.title()} Value", val_list, key=f"sb_attr_{attr_name}_{key_suffix}")
                if selected_val:
                    st.markdown(f"**Value:** `{selected_val}`")
                    _draw_kw_table(groups[selected_val], f"{attr_name}={selected_val}", f"a_{attr_name}_{selected_val}_{key_suffix}")

