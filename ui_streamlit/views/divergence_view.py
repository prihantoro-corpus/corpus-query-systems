import streamlit as st
import pandas as pd
# Trigger hot reload
from ui_streamlit.state_manager import get_state, set_state
from core.modules.divergence import calculate_divergence_for_nodes
from core.io_utils import df_to_excel_bytes
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions
import time

def render_divergence_view():
    st.markdown("## 🔍 Divergence Analysis (Sub-Corpora Comparison)")
    st.error("🚨 **THIS FEATURE IS STILL IN DEVELOPMENT. STILL UNSTABLE** 🚨")
    
    with st.expander("💡 **Method & Transparency: Divergence Analysis (Surprise vs Core Words)**", expanded=False):
        st.markdown("""
        **Goal:** Identify how the meaning and framing of words change across different sub-corpora. 
        
        Instead of just telling you *how often* a word is used, it looks at a word's "company" (its collocates) to reveal:
        *   **Surprise Words (High Divergence):** Words whose framing is fiercely contested or completely different between your sub-corpora.
        *   **Core Words (High Convergence):** Words whose framing is stable, universal, and agreed-upon across all your sub-corpora.
        
        **Statistical Measures:**
        *   **Generalized Jaccard:** A strict, binary check of shared vocabulary.
        *   **Jensen-Shannon Divergence (JSD):** A robust method comparing the actual *strength and probability* of collocates mathematically.
        """)
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Main Corpus')
    corpus_stats = get_state('corpus_stats')
    
    if not corpus_path:
        st.warning("⚠️ Please load a Corpus in the sidebar Overview to use Divergence Analysis.")
        return

    st.markdown("### 1. Scope & Node Settings")
    
    node_mode = st.radio("Node Selection Mode", ["Auto-Discover (Corpus-Driven)", "Manual Node Words"], horizontal=True)
    
    if node_mode == "Manual Node Words":
        nodes_input = st.text_input("Node Words (comma-separated)", value="training, blame, cash, jobs, diabetes", help="Enter the words you want to calculate divergence for.")
    else:
        st.info("Auto-Discover automatically finds the most divergent words by scanning the most frequent vocabulary in your corpus.")
        col_ad1, col_ad2, col_ad3 = st.columns(3)
        with col_ad1:
            auto_min_freq = st.number_input("Minimum Node Frequency", min_value=1, value=20, help="Only analyze words that appear at least this many times in the main corpus.")
        with col_ad2:
            auto_pos_filter = st.text_input("POS Filter (Optional)", value="NN|NP|V|JJ|RB", help="Regex substring for POS tags (e.g., NN|V). Matches any tag containing these letters.")
        with col_ad3:
            auto_limit = st.selectbox("Max Words to Analyze", [100, 500, 1000, 2000, 5000, "All Eligible"], index=2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 2. Collocation Rules")
        window_size = st.number_input("Window Size (Left & Right)", min_value=1, max_value=20, value=5)
        assoc_measure = st.selectbox("Association Measure", ["Log-Likelihood", "MI", "Dice", "Log-Dice"], index=0)
        min_pair_freq = st.number_input("Minimum Pair Frequency", min_value=1, value=5)
        min_colloc_freq = st.number_input("Minimum Overall Collocate Frequency", min_value=1, value=10, help="Filters out ultra-rare words.")
        
        st.markdown("**Advanced Filters**")
        pos_filter = st.text_input("POS Filter (Include/Exclude)", value="NN*|NP*|V*|JJ*|RB*", help="Filters collocates. Default allows only Nouns, Verbs, Adjectives, Adverbs (naturally removes stopwords).")
        token_filter = st.text_input("Token Filter", value="", help="e.g., -(the,and,of). Leave blank if using the POS filter above.")

    with col2:
        st.markdown("### 3. Divergence Math Settings")
        div_metric = st.selectbox("Divergence Statistic", ["Jensen-Shannon Divergence", "Generalized Jaccard"], index=0)
        top_k = st.selectbox("Top-K Calculation Threshold", [10, 20, 30, 50, 100], index=1, help="Calculate divergence based on the Top N strongest collocates.")
        min_node_elig = st.number_input("Minimum Node Eligibility", min_value=1, value=10, help="Node must have at least this many valid collocates in ALL sub-corpora.")
        
    st.markdown("---")
    st.markdown("### 4. Define Sub-Corpora (XML Metadata)")
    
    if 'div_num_subcorpora' not in st.session_state:
        st.session_state['div_num_subcorpora'] = 2

    # Dynamic columns for sub-corpora
    num_sc = st.session_state['div_num_subcorpora']
    cols = st.columns(min(num_sc, 4)) # Wrap if more than 4, or just render in rows of 2-3
    
    corpora_details_inputs = []
    
    for i in range(num_sc):
        col_idx = i % 4
        if i > 0 and col_idx == 0:
            cols = st.columns(4) # start a new row of columns
            
        with cols[col_idx]:
            st.markdown(f"**Sub-Corpus {i+1}**")
            sc_name = st.text_input(f"Label for Sub-Corpus {i+1}", value=f"Sub-Corpus {i+1}", key=f"div_sc_name_{i}")
            xml_filters = render_xml_restriction_filters(corpus_path, f"div_sc{i}", corpus_name=corpus_name)
            xml_where, xml_params = apply_xml_restrictions(xml_filters)
            
            corpora_details_inputs.append({
                'name': sc_name,
                'xml_where': xml_where,
                'xml_params': xml_params
            })
            st.markdown("<br>", unsafe_allow_html=True)
            
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ Add Sub-Corpus", help="Add another sub-corpus to compare."):
            st.session_state['div_num_subcorpora'] += 1
            st.rerun()
    with col_btn2:
        if num_sc > 2:
            if st.button("➖ Remove Sub-Corpus", help="Remove the last sub-corpus."):
                st.session_state['div_num_subcorpora'] -= 1
                st.rerun()

    st.markdown("---")
    st.warning("⚠️ **Note:** Divergence analysis is computationally intensive. For large vocabularies or many sub-corpora, this may take several minutes.")
    
    if st.button("Calculate Divergence", type="primary", use_container_width=True):
        if node_mode == "Manual Node Words":
            nodes_list = [n.strip() for n in nodes_input.split(",") if n.strip()]
            if not nodes_list:
                st.error("Please enter at least one node word.")
                return
        else:
            import duckdb
            try:
                con = duckdb.connect(corpus_path, read_only=True)
                cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
                has_lemma = any(c[1].lower() == 'lemma' for c in cols_info)
                node_col = "lower(lemma)" if has_lemma else "_token_low"
                
                pos_where = ""
                if auto_pos_filter:
                    clean_pattern = auto_pos_filter.replace('_', '')
                    pos_where = f" AND regexp_matches(pos, '{clean_pattern}')"
                        
                limit_sql = "" if auto_limit == "All Eligible" else f"LIMIT {auto_limit}"
                
                query = f"""
                SELECT {node_col} as l
                FROM corpus 
                WHERE 1=1 {pos_where}
                GROUP BY l 
                HAVING COUNT(*) >= {auto_min_freq}
                ORDER BY COUNT(*) DESC
                {limit_sql}
                """
                nodes_list = [row[0] for row in con.execute(query).fetchall() if row[0]]
                con.close()
                
                if not nodes_list:
                    st.error("Auto-Discover found 0 eligible words. Try lowering the frequency or adjusting the POS filter.")
                    return
                st.info(f"Auto-Discover scanning {len(nodes_list)} words...")
            except Exception as e:
                st.error(f"Error fetching auto-discover words: {e}")
                return
            
        corpora_details = []
        for details in corpora_details_inputs:
            corpora_details.append({
                'path': corpus_path,
                'name': details['name'],
                'xml_where': details['xml_where'],
                'xml_params': details['xml_params'],
                'stats': corpus_stats
            })
        
        start_time = time.time()
        
        with st.spinner(f"Crunching divergence scores across {num_sc} sub-corpora..."):
            try:
                df_results = calculate_divergence_for_nodes(
                    corpora_details=corpora_details,
                    nodes_list=nodes_list,
                    window_size=window_size,
                    assoc_measure=assoc_measure,
                    min_pair_freq=min_pair_freq,
                    min_collocate_freq=min_colloc_freq,
                    top_k=top_k,
                    metric=div_metric,
                    is_raw_mode=get_state('is_raw_mode', False),
                    token_filter=token_filter,
                    pos_filter=pos_filter,
                    lemma_filter="",
                    min_node_collocates=min_node_elig
                )
                
                if df_results.empty:
                    st.error("🐢 **Apologies, no results found.**")
                    st.info("💡 **Tip:** Try reducing the 'Top-K Threshold', increasing 'Minimum Frequencies', or ensuring the words exist in all sub-corpora.")
                else:
                    st.success(f"Analysis complete in {round(time.time() - start_time, 2)} seconds!")
                    set_state('divergence_results', df_results)
                    
            except Exception as e:
                st.error(f"💾 **Apologies, an error occurred during calculation:** {str(e)}")
                st.info("💡 **Tip:** Please try applying stricter filters to reduce the vocabulary size.")

    # Render Results if they exist
    if 'divergence_results' in st.session_state and not st.session_state['divergence_results'].empty:
        df = st.session_state['divergence_results']
        
        st.markdown("### 5. Results")
        
        col_disp, col_exp = st.columns([3,1])
        with col_disp:
            display_limit = st.selectbox("Collocates to Display per Corpus (Visual Only)", ["5", "10", "20", "All"], index=0)
        
        def format_collocates(text_list_str, limit_str):
            if limit_str == "All" or pd.isna(text_list_str) or not text_list_str:
                return text_list_str
            items = [x.strip() for x in text_list_str.split(",")]
            limit = int(limit_str)
            return ", ".join(items[:limit])

        # Apply visual limit
        df_display = df.copy()
        for col in df_display.columns:
            if 'Collocates' in col:
                df_display[col] = df_display[col].apply(lambda x: format_collocates(x, display_limit))
                
        # Split into Divergent (Surprise) and Convergent (Core)
        # Sort by Divergence Score descending
        df_surprise = df_display.sort_values(by="Divergence Score", ascending=False)
        # Sort by Divergence Score ascending
        df_core = df_display.sort_values(by="Divergence Score", ascending=True)

        st.markdown("#### Table 1: Highest Divergence (The 'Surprise Words')")
        st.dataframe(df_surprise, use_container_width=True, hide_index=True)
        
        st.markdown("#### Table 2: Highest Convergence (The 'Core Words')")
        st.dataframe(df_core, use_container_width=True, hide_index=True)
        
        with col_exp:
            st.markdown("<br>", unsafe_allow_html=True)
            excel_data = df_to_excel_bytes(df)
            st.download_button(
                label="📥 Download to Excel",
                data=excel_data,
                file_name="divergence_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
