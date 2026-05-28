import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
import core.modules.overview as ov
import importlib
importlib.reload(ov)
from core.ai_service import interpret_results_llm, guess_pos_definitions
from core.preprocessing.xml_parser import format_structure_data_hierarchical, apply_xml_restrictions
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.visualiser.wordcloud import create_word_cloud
from core.io_utils import df_to_excel_bytes
import duckdb
from core.modules.classification import (
    classify_sentiment_vader, 
    classify_topics_keyword_weighted, 
    classify_topics_bertopic,
    apply_classification_by_sentence,
    BERTOPIC_AVAILABLE
)
import core.modules.readability as rd
import importlib
importlib.reload(rd)

get_sentence_stats = rd.get_sentence_stats
compute_readability_metrics = rd.compute_readability_metrics
apply_reading_ease_annotation = rd.apply_reading_ease_annotation
annotate_reading_ease_by_chunks = rd.annotate_reading_ease_by_chunks
get_chunk_readability_stats = rd.get_chunk_readability_stats
calculate_formulas = rd.calculate_formulas
map_score_to_level = rd.map_score_to_level

# Standard Language Mapping
LANG_MAP = {
    "EN": "English",
    "ID": "Indonesian",
    "AR": "Arabic",
    "JP": "Japanese",
    "CH": "Chinese",
    "KO": "Korean",
    "LO": "Limola"
}

def _render_language_confirmation(path, key_suffix=""):
    """
    Renders the language selection and confirmation UI.
    """
    current_lang = ov.get_corpus_language(path)
    
    # Try to find the code from the full name if stored that way
    current_code = "EN"
    for code, name in LANG_MAP.items():
        if current_lang == name or current_lang == code:
            current_code = code
            break

    with st.expander("🌐 Corpus Language Settings"):
        st.caption("Confirm the language of this corpus to enable dictionary and thesaurus links.")
        
        # Show currently confirmed language
        st.info(f"**Currently Confirmed:** {current_code} - {LANG_MAP.get(current_code, 'English')}")
        
        c_lang1, c_lang2 = st.columns([3, 1])
        
        lang_options = [f"{code} - {name}" for code, name in LANG_MAP.items()]
        try:
            current_idx = list(LANG_MAP.keys()).index(current_code)
        except ValueError:
            current_idx = 0
            
        with c_lang1:
            selected_fmt = st.selectbox(
                "Language", 
                lang_options,
                index=current_idx,
                key=f"lang_select_{key_suffix}",
                label_visibility="collapsed"
            )
            selected_code = selected_fmt.split(" - ")[0]
            selected_name = LANG_MAP[selected_code]

        with c_lang2:
            if st.button("Confirm", key=f"lang_confirm_{key_suffix}", use_container_width=True):
                if ov.set_corpus_language(path, selected_name):
                    set_state('target_lang', selected_code)
                    st.toast(f"✅ {selected_code} Confirmed!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to save language.")
    
    set_state(f'current_language_{key_suffix}', selected_name)

def render_overview():
    st.header("Corpus Overview")
    
    comp_mode = get_state('comparison_mode', False)
    
    if not comp_mode:
        # Standard Single View
        corpus_path = get_state('current_corpus_path')
        if not corpus_path:
            source_type = get_state('source_type')
            if source_type == "Online Corpus":
                render_online_builder_ui()
            else:
                st.info("Please load a corpus from the sidebar to view statistics.")
            return
        stats = get_state('corpus_stats')
        name = get_state('current_corpus_name')
        structure = get_state('xml_structure_data')
        error = get_state('xml_structure_error')
        render_full_overview(name, corpus_path, stats, structure, error)
    else:
        # Comparison Side-by-Side
        c1_path = get_state('current_corpus_path')
        c2_path = get_state('comp_corpus_path')
        
        if not c1_path and not c2_path:
            st.info("Please load corpora to compare.")
            return
            
        col_a, col_b = st.columns(2)
        
        with col_a:
            if c1_path:
                render_overview_stats(
                    get_state('current_corpus_name'),
                    c1_path,
                    get_state('corpus_stats'),
                    get_state('xml_structure_data'),
                    get_state('xml_structure_error'),
                    key_suffix="c1"
                )
            else:
                st.warning("Primary Corpus not loaded.")
                
        with col_b:
            if c2_path:
                render_overview_stats(
                    get_state('comp_corpus_name'),
                    c2_path,
                    get_state('comp_corpus_stats'),
                    get_state('comp_xml_structure_data'),
                    None, # Error for comp?
                    key_suffix="c2"
                )
            else:
                st.warning("Comparison Corpus not loaded.")

def render_overview_stats(name, path, stats, structure, error, key_suffix=""):
    st.subheader(f"📊 {name}")
    
    # --- XML Restriction Filters ---
    xml_filters = render_xml_restriction_filters(path, f"overview_{key_suffix}")
    xml_where, xml_params = apply_xml_restrictions(xml_filters)
    
    # Use restricted stats if filters are active
    if xml_filters:
        display_stats = ov.get_restricted_stats(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        display_stats = ov.calculate_corpus_statistics(stats, db_path=path)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens", f"{display_stats.get('total_tokens', 0):,}")
    m2.metric("Types", f"{display_stats.get('unique_types', 0):,}")
    m3.metric("TTR", f"{display_stats.get('ttr', 0):.4f}")

    # Language Settings removed: choosing is now automatic or sidebar-driven
    # _render_language_confirmation(path, key_suffix)


    # Show classification for ALL languages now (via Translation)
    show_classification = True
    
    tabs_list = ["XML", "Sub-corpus Stats", "Freq", "POS", "Cloud", "Metadata", "🏷️ Sentiment & Topic", "📖 Reading Ease"]
    
    tabs = st.tabs(tabs_list)
    
    # Unpack tabs
    tab1, tab_sub, tab2, tab3, tab4, tab_meta, tab_cls, tab_re = tabs
        
    with tab1:
        if error: st.error(error)
        if structure:
            html = format_structure_data_hierarchical(structure)
            st.markdown(f'<div style="font-family: monospace; font-size: 0.85em; padding: 10px; background: #1e1e1e; border-radius: 5px; color: #d4d4d4;">{html}</div>', unsafe_allow_html=True)
        else: st.caption("No XML structure.")

    with tab_sub:
        _render_subcorpus_stats(path, key_suffix)
        
    with tab2:
        df = ov.get_top_frequencies_v2(path, limit=50, xml_where_clause=xml_where, xml_params=xml_params)
        if not df.empty:
            # Use restricted total for PMW calculation
            total = display_stats.get('total_tokens', 1)
            df['Rel. Freq'] = (df['frequency'] / total * 1_000_000).round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else: st.caption("No frequencies.")
        
    with tab3:
        _render_pos_management_tab(path, xml_where, xml_params, key_suffix)
        
    with tab4:
        f_df = ov.get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not f_df.empty:
            fig = create_word_cloud(f_df, 'pos' in f_df.columns)
            if fig: st.pyplot(fig)
        else: st.caption("No wordcloud.")

    with tab_meta:
        _render_metadata_annotation_tab(path, key_suffix)
        
    if tab_cls:
        with tab_cls:
            _render_classification_tab(path, key_suffix)

    with tab_re:
        _render_reading_ease_tab(path, key_suffix)

def render_full_overview(name, path, stats, structure, error):
    # --- XML Restriction Filters ---
    xml_filters = render_xml_restriction_filters(path, "overview_full")
    xml_where, xml_params = apply_xml_restrictions(xml_filters)

    # Use restricted stats if filters are active
    if xml_filters:
        display_stats = ov.get_restricted_stats(path, xml_where_clause=xml_where, xml_params=xml_params)
    else:
        display_stats = ov.calculate_corpus_statistics(stats, db_path=path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tokens", f"{display_stats.get('total_tokens', 0):,}")
    col2.metric("Unique Types", f"{display_stats.get('unique_types', 0):,}")
    col3.metric("Type/Token Ratio (TTR)", f"{display_stats.get('ttr', 0):.4f}")
    
    # Language Confirmation removed
    # _render_language_confirmation(path, "full")
        
    st.markdown("---")
    

    current_lang = ov.get_corpus_language(path)
    show_classification = True
    
    tabs_list = ["XML Structure", "Sub-corpus Stats", "Top Frequencies", "Unique POS Tags", "Word Cloud", "Metadata Annotation", "🏷️ Sentiment & Topic Analysis", "📖 Reading Ease"]

    tabs = st.tabs(tabs_list)
    tab1, tab_sub, tab2, tab3, tab4, tab_meta, tab_cls, tab_re = tabs
    
    with tab1:
        if error: st.error(error)
        if structure:
            st.subheader("Structure and Attributes")
            html = format_structure_data_hierarchical(structure)
            st.markdown(f'<div style="font-family: monospace; font-size: 0.9em; padding: 15px; background: #1e1e1e; border-radius: 8px; color: #d4d4d4;">{html}</div>', unsafe_allow_html=True)
            
            with st.expander("Show Raw Python Data (for diagnosis)"):
                 st.info("The data below is the Python dictionary successfully produced by the XML parser.")
                 st.json(structure)

            with st.expander("Database Diagnostics"):
                import duckdb
                st.write(f"DB Path: `{path}`")
                try:
                    c = duckdb.connect(path, read_only=True)
                    info = c.execute("PRAGMA table_info(corpus)").fetch_df()
                    st.write("Table Schema:", info)
                    
                    # Columns check
                    cols = info['name'].tolist()
                    standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename'}
                    meta = [c for c in cols if c not in standard]
                    st.write("Detected Metadata Columns:", meta)
                    
                    if meta:
                        rows = c.execute(f"SELECT {', '.join(meta)} FROM corpus LIMIT 5").fetch_df()
                        st.write("Sample Metadata:", rows)
                    c.close()
                except Exception as e:
                    st.error(str(e))
        else: st.info("No XML structure metadata available.")

    with tab_sub:
        _render_subcorpus_stats(path, "full")
            
    with tab2:
        st.subheader("Top Frequency Tokens")
        df = ov.get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not df.empty:
            # Use restricted total for PMW calculation
            total = display_stats.get('total_tokens', 1)
            df['Rel. Freq (per M)'] = (df['frequency'] / total * 1_000_000).round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download Top 100", data=df_to_excel_bytes(df), file_name=f"{name}_top_freq.xlsx")
        else: st.info("No frequency data.")

    with tab3:
        st.subheader("Unique POS Tags (and Definitions)")
        _render_pos_management_tab(path, xml_where, xml_params, "full")

    with tab4:
        st.subheader("Word Cloud")
        f_df = ov.get_top_frequencies_v2(path, limit=100, xml_where_clause=xml_where, xml_params=xml_params)
        if not f_df.empty:
            fig = create_word_cloud(f_df, 'pos' in f_df.columns)
            if fig:
                if 'pos' in f_df.columns:
                     st.markdown('<div style="font-size: 0.8em; margin-bottom: 5px;"><span style="color:#33CC33;">●</span> Noun | <span style="color:#3366FF;">●</span> Verb | <span style="color:#FF33B5;">●</span> Adj | <span style="color:#FFCC00;">●</span> Adv</div>', unsafe_allow_html=True)
                st.pyplot(fig)
        else: st.info("No frequency data.")

    with tab_meta:
        _render_metadata_annotation_tab(path, "full")

    if tab_cls:
        with tab_cls:
            _render_classification_tab(path, "full")

    with tab_re:
        _render_reading_ease_tab(path, "full")

    st.markdown("---")
    if st.button("🧠 Interpret Corpus Overview (LLM)", key="llm_overview_btn"):
        with st.spinner("AI is analyzing..."):
            overview_data = {"stats": display_stats, "top10": df.head(10).to_dict(orient='records') if not df.empty else {}}
            resp, err = interpret_results_llm(
                target_word=name, 
                analysis_type="Corpus Overview", 
                data_description="Stats and Freq", 
                data=str(overview_data),
                ai_provider=get_state('ai_provider'),
                gemini_api_key=get_state('gemini_api_key'),
                ollama_url=get_state('ollama_url'),
                ollama_model=get_state('ai_model')
            )
            if resp:
                set_state('llm_res_overview', resp)
            elif err:
                st.error(err)
            
    llm_res = get_state('llm_res_overview')
    if llm_res:
        with st.expander("🤖 AI Assistant Interpretation", expanded=True):
            st.markdown(llm_res)

def _render_pos_management_tab(path, xml_where, xml_params, key_suffix):
    """
    Helper to render the POS management tab content.
    """
    tags = ov.get_unique_pos_tags(path, xml_where_clause=xml_where, xml_params=xml_params)
    
    if tags:
        # Load definitions
        current_defs = ov.get_pos_definitions(path)
        
        # Prepare DataFrame
        data_rows = []
        for t in tags:
            data_rows.append({"Tag": t, "Definition": current_defs.get(t, "")})
        df_tags = pd.DataFrame(data_rows)
        
        st.info("Edit POS definitions. Use AI to guess, upload a file, or edit the table below.")
        
        # --- ACTION BUTTONS ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            if st.button("✨ AI Guess Definitions", key=f"ai_guess_pos_{key_suffix}"):
                with st.spinner("AI is guessing definitions..."):
                    guesses, err = guess_pos_definitions(
                        tags, 
                        ai_provider=get_state('ai_provider'),
                        gemini_api_key=get_state('gemini_api_key'),
                        ollama_url=get_state('ollama_url'),
                        ollama_model=get_state('ai_model')
                    )
                    if guesses:
                        for t in tags:
                            if t in guesses:
                                current_defs[t] = guesses[t]
                        set_state(f'temp_pos_defs_{path}', current_defs)
                        st.success("AI Guesses Applied! Review and Save.")
                        st.rerun()
                    else:
                        st.error(err or "AI provided no guesses.")

        with c2:
            with st.popover("📂 Upload / Parse", use_container_width=True):
                st.markdown("### Import Definitions")
                st.markdown("Format: `TAG: Definition` (one per line)")
                
                uploaded = st.file_uploader("Upload Text File", type=['txt'], key=f"pos_upload_{key_suffix}")
                if uploaded:
                    content = uploaded.read().decode('utf-8')
                    full_text_input = content
                else:
                    full_text_input = ""
                    
                text_input = st.text_area("Or Paste Here", value=full_text_input, height=150, key=f"pos_paste_{key_suffix}")
                
                if st.button("Process Input", key=f"pos_process_{key_suffix}"):
                    count = 0
                    for line in text_input.split('\n'):
                        line = line.strip()
                        if not line: continue
                        
                        parts = None
                        if '\t' in line:
                            parts = line.split('\t', 1)
                        elif ':' in line:
                            parts = line.split(':', 1)
                        
                        if parts:
                            t_key = parts[0].strip()
                            t_val = parts[1].strip()
                            if t_key in tags:
                                current_defs[t_key] = t_val
                                count += 1
                    
                    set_state(f'temp_pos_defs_{path}', current_defs)
                    st.success(f"Parsed {count} definitions.")
                    st.rerun()

        # --- EDITOR ---
        temp_defs = get_state(f'temp_pos_defs_{path}')
        if temp_defs:
            data_rows = [{"Tag": t, "Definition": temp_defs.get(t, "")} for t in tags]
            df_tags = pd.DataFrame(data_rows)
        
        edited_df = st.data_editor(
            df_tags, 
            key=f"pos_editor_{key_suffix}", 
            hide_index=True, 
            use_container_width=True,
            disabled=["Tag"]
        )
        
        if st.button("💾 Save Definitions", key=f"save_pos_{key_suffix}", type="primary", use_container_width=True):
            new_defs = dict(zip(edited_df['Tag'], edited_df['Definition']))
            if ov.save_pos_definitions(path, new_defs):
                st.toast("Definitions Saved!", icon="✅")
                set_state(f'temp_pos_defs_{path}', None)
                st.rerun()
            else:
                st.error("Failed to save.")

    else:
        st.info("No POS tags detected.")

def _render_classification_tab(db_path, key_suffix):
    """
    Renders the Topic & Sentiment Labeling UI with method selection.
    """
    st.markdown("#### 🏷️ Automatic Corpus Labeling")
    
    with st.expander("💡 **Method & Transparency: Classification**", expanded=False):
        st.markdown("""
        **Sentiment Analysis:** Uses the VADER lexicon to score sentences as Positive, Negative, or Neutral.
        
        **Topic Classification:**
        - **TF-IDF (Fast):** Uses pre-defined keywords to categorize text into standard topics like Sport, Politics, etc.
        - **BERTopic (Accurate):** Uses advanced embedding models to automatically discover "natural" topics in your specific corpus.
        
        **Editability:** You can rename topics or adjust keywords in the results preview before applying them.
        """)
        
    st.caption("Automatically tag sentences with **Sentiment** and **Topic** using local NLP libraries.")
    
    # Check Columns
    try:
        con = duckdb.connect(db_path, read_only=True)
        cols = [c[1] for c in con.execute("PRAGMA table_info(corpus)").fetchall()]
        has_topic = 'topic' in cols
        has_sent = 'sentiment' in cols
        con.close()
        
        found_labels = []
        if has_topic: found_labels.append("Topic")
        if has_sent: found_labels.append("Sentiment")
        
        if found_labels:
            st.success(f"✅ Existing labels found: {', '.join(found_labels)}")
        else:
            st.info("No sentiment or topic labels found yet. Configure and run labeling below.")
    except Exception as e: 
        # st.error(f"Debug: {e}")
        pass

    # Non-English Sentiment Warning
    curr_lang = ov.get_corpus_language(db_path)
    if curr_lang and curr_lang.lower() not in ['en', 'english']:
        st.warning("⚠️ **Non-English Sentiment Analysis:** Sentences will be translated to English first. This may take significant time for large corpora and may hit translation limits.")

    st.markdown("---")
    
    # Method Selection
    method = st.radio(
        "**Topic Classification Method:**",
        options=["TF-IDF (Fast)", "BERTopic (Accurate)"],
        horizontal=True,
        help="TF-IDF: Pre-defined topics, instant. BERTopic: Auto-discovered, requires 500MB model download.",
        key=f"topic_method_{key_suffix}"
    )
    
    use_bertopic = "BERTopic" in method
    
    # Show method-specific info
    if use_bertopic:
        st.warning("⚠️ **BERTopic requires ~500MB model download** and longer processing time, but provides more accurate results.")
        
        with st.expander("🛠️ BERTopic Technical Details"):
            st.info("""
            **No data is sent to external AI servers.** All processing happens locally:
            - Uses [BERTopic](https://github.com/MaartenGr/BERTopic) for topic modeling
            - Downloads sentence-transformers model (all-MiniLM-L6-v2) on first use
            - Automatically discovers topics from your corpus content
            """)
    else:
        with st.expander("🛠️ TF-IDF Technical Details"):
            st.info("""
            **No data is sent to external AI servers.** All processing happens locally:
            - **Sentiment Analysis**: Uses [NLTK VADER](https://github.com/cjhutto/vaderSentiment) (Rule-based).
            - **Multi-language**: Non-English text is automatically translated to English for sentiment analysis.
            - **Topic Classification**: Uses [Scikit-learn](https://scikit-learn.org/) TF-IDF with pre-defined keyword categories.
            """)
    
    st.markdown("---")
    
    # Configuration Section
    st.write("**Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        do_sent = st.checkbox("Sentiment (Pos/Neg/Neu)", value=True, key=f"chk_sent_{key_suffix}")
    
    with col2:
        do_topic = st.checkbox("Topic Classification", value=True, key=f"chk_topic_{key_suffix}")
    
    # BERTopic-specific parameters
    if use_bertopic and do_topic:
        st.write("**BERTopic Parameters:**")
        st.caption("💡 Tip: Use fewer topics (8-12) for clearer categorization. Higher min topic size reduces noise.")
        
        bcol1, bcol2 = st.columns(2)
        
        with bcol1:
            n_topics_option = st.selectbox(
                "Number of Topics",
                options=[8, 10, 12, 15, "Auto"] + list(range(5, 21)),
                index=1,  # Default to 10
                key=f"bertopic_n_topics_{key_suffix}",
                help="Recommended: 8-12 topics. Auto may create too many."
            )
            n_topics = 'auto' if n_topics_option == "Auto" else n_topics_option
        
        with bcol2:
            min_topic_size = st.number_input(
                "Min Topic Size",
                min_value=5,
                max_value=100,
                value=20,  # Increased from 10 to reduce topic count
                key=f"bertopic_min_size_{key_suffix}",
                help="Higher values = fewer, more distinct topics"
            )
    
    # Run Labeling Button
    if st.button("🚀 Run Labeling", key=f"run_cls_{key_suffix}", disabled=not (do_sent or do_topic)):
        with st.spinner("Processing sentences..."):
            try:
                con = duckdb.connect(db_path, read_only=True)
                df_sents = con.execute("""
                    SELECT filename, sent_id, string_agg(token, ' ' ORDER BY id) as text 
                    FROM corpus 
                    GROUP BY filename, sent_id
                """).fetch_df()
                con.close()
                
                if df_sents.empty:
                    st.error("Corpus is empty.")
                    return

                texts = df_sents['text'].tolist()
                
                # Sentiment Analysis
                if do_sent:
                    st.write("Computing Sentiment...")
                    # Get current language from DB or State
                    lang_for_sent = ov.get_corpus_language(db_path)
                    df_sents['Predicted Sentiment'] = classify_sentiment_vader(texts, lang=lang_for_sent)
                
                # Topic Classification
                topic_info = None
                if do_topic:
                    if use_bertopic:
                        st.write("Computing Topics with BERTopic (this may take a while)...")
                        
                        if not BERTOPIC_AVAILABLE:
                            st.error("BERTopic is not installed. Please run: `pip install bertopic sentence-transformers`")
                            return
                        
                        topic_assignments, topic_info = classify_topics_bertopic(
                            texts, 
                            n_topics=n_topics,
                            min_topic_size=min_topic_size
                        )
                        df_sents['Predicted Topic'] = topic_assignments
                    else:
                        st.write("Computing Topics with TF-IDF...")
                        topic_assignments, topic_info = classify_topics_keyword_weighted(texts)
                        df_sents['Predicted Topic'] = topic_assignments
                
                # Store results
                set_state(f'cls_preview_{key_suffix}', df_sents)
                if topic_info:
                    set_state(f'cls_topic_info_{key_suffix}', topic_info)
                
                st.toast("Labeling Complete! Preview below.", icon="🎉")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Preview & Edit Section
    preview_df = get_state(f'cls_preview_{key_suffix}')
    topic_info = get_state(f'cls_topic_info_{key_suffix}')
    
    if preview_df is not None:
        st.divider()
        st.subheader("Preview & Edit Labels")
        
        # Topic Info Editor (if topics were generated)
        if topic_info and 'Predicted Topic' in preview_df.columns:
            st.write("**Edit Topic Labels & Keywords:**")
            st.caption("Customize the topic names and keywords before applying to corpus.")
            
            # Build editable dataframe
            topic_rows = []
            for topic_key, info in topic_info.items():
                topic_rows.append({
                    'Topic ID': str(topic_key),
                    'Label': info['label'],
                    'Keywords': ', '.join(info['keywords']) if info['keywords'] else '',
                    'Count': info['count']
                })
            
            topic_edit_df = pd.DataFrame(topic_rows)
            
            edited_topics = st.data_editor(
                topic_edit_df,
                key=f"topic_editor_{key_suffix}",
                hide_index=True,
                use_container_width=True,
                disabled=['Topic ID', 'Count'],
                column_config={
                    'Label': st.column_config.TextColumn('Topic Label', width='medium'),
                    'Keywords': st.column_config.TextColumn('Keywords (comma-separated)', width='large'),
                }
            )
            
            # Update topic assignments based on edits
            if not edited_topics.equals(topic_edit_df):
                # Create mapping from old labels to new labels
                label_map = {}
                for idx, row in edited_topics.iterrows():
                    old_label = topic_edit_df.iloc[idx]['Label']
                    new_label = row['Label']
                    label_map[old_label] = new_label
                
                # Apply mapping to preview_df
                preview_df['Predicted Topic'] = preview_df['Predicted Topic'].map(
                    lambda x: label_map.get(x, x)
                )
                set_state(f'cls_preview_{key_suffix}', preview_df)
        
        # Sample Preview
        st.write("**Sample Sentences:**")
        show_cols = ['text']
        if 'Predicted Topic' in preview_df.columns: show_cols.append('Predicted Topic')
        if 'Predicted Sentiment' in preview_df.columns: show_cols.append('Predicted Sentiment')
        
        st.dataframe(preview_df[show_cols].head(20), use_container_width=True)
        
        # Apply to Corpus
        save_col1, save_col2 = st.columns([1, 1])
        with save_col1:
            st.warning("⚠️ This will modify the corpus database. User consent is required to apply these changes.")
        with save_col2:
            if st.button("✅ I Agree, Apply to Corpus", key=f"save_cls_{key_suffix}", type="primary"):
                with st.spinner("Updating database..."):
                    success = apply_classification_by_sentence(
                        db_path, 
                        preview_df['filename'].tolist(),
                        preview_df['sent_id'].tolist(),
                        topics=preview_df['Predicted Topic'].tolist() if 'Predicted Topic' in preview_df.columns else None,
                        sentiments=preview_df['Predicted Sentiment'].tolist() if 'Predicted Sentiment' in preview_df.columns else None
                    )
                    
                    if success:
                        st.success("Corpus updated successfully!")
                        set_state(f'cls_preview_{key_suffix}', None)
                        set_state(f'cls_topic_info_{key_suffix}', None)
                        st.toast("Applied! Refreshing...", icon="💾")
                        st.rerun()
                    else:
                        st.error("Database update failed.")

def _render_subcorpus_stats(db_path, key_suffix=""):
    """
    Renders charts and tables for sub-corpus statistics:
    1. By File Name
    2. By Topic & Sentiment (if available)
    3. By XML Attributes (if available)
    """
    import plotly.express as px
    
    st.subheader("Sub-Corpus Statistics")
    
    conn = duckdb.connect(db_path, read_only=True)
    try:
        # 1. By File Name
        st.markdown("##### 📂 By File Name")
        df_files = conn.execute("""
            SELECT 
                filename, 
                COUNT(*) as Tokens,
                CAST(COUNT(DISTINCT _token_low) AS FLOAT) / COUNT(*) as TTR
            FROM corpus 
            GROUP BY filename 
            ORDER BY Tokens DESC
        """).fetch_df()
        
        if not df_files.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                # Use Bar Chart for files as there might be many
                fig = px.bar(df_files, x='filename', y='Tokens', title="Tokens per File")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(
                    df_files.style.format({'TTR': '{:.4f}'}), 
                    use_container_width=True, 
                    hide_index=True
                )
        else:
            st.info("No file information available.")
            
        st.divider()
        
        # 2. By Topic & Sentiment
        cols_info = conn.execute("PRAGMA table_info(corpus)").fetch_df()
        cols = [c.lower() for c in cols_info['name'].tolist()]
        
        has_topic = 'topic' in cols
        has_sent = 'sentiment' in cols
        
        if has_topic or has_sent:
            st.markdown("##### 🏷️ By Automatic Labeling")
            
            if has_topic:
                 # Group by distinct Topic (handling NULLs)
                topic_data = conn.execute("SELECT topic, COUNT(*) as Count FROM corpus WHERE topic IS NOT NULL GROUP BY topic ORDER BY Count DESC").fetch_df()
                if not topic_data.empty:
                    st.write("**Topic Distribution**")
                    tc1, tc2 = st.columns([1, 1])
                    with tc1:
                        fig_t = px.pie(topic_data, names='topic', values='Count', title="Topic Distribution")
                        st.plotly_chart(fig_t, use_container_width=True)
                    with tc2:
                         st.dataframe(topic_data, use_container_width=True, hide_index=True)
                else:
                    st.info("Topic column exists but no topics found. Run 'Automatic Labeling'.")

            if has_sent:
                # Group by distinct Sentiment
                sent_data = conn.execute("SELECT sentiment, COUNT(*) as Count FROM corpus WHERE sentiment IS NOT NULL GROUP BY sentiment ORDER BY Count DESC").fetch_df()
                if not sent_data.empty:
                    st.write("**Sentiment Distribution**")
                    sc1, sc2 = st.columns([1, 1])
                    with sc1:
                        fig_s = px.pie(sent_data, names='sentiment', values='Count', title="Sentiment Distribution", 
                                       color='sentiment', 
                                       color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'})
                        st.plotly_chart(fig_s, use_container_width=True)
                    with sc2:
                         st.dataframe(sent_data, use_container_width=True, hide_index=True)
                else:
                    st.info("Sentiment column exists but no sentiments found. Run 'Automatic Labeling'.")
        else:
            st.info("No Topic/Sentiment labels found. Go to the 'Automatic Labeling' tab to generate them.")
            
        st.divider()

        # 3. By XML Attributes
        from core.preprocessing.xml_parser import get_xml_attribute_columns
        attr_cols = get_xml_attribute_columns(conn)
        
        if attr_cols:
            st.markdown("##### 🧱 By XML Attributes")
            st.caption("Distribution of tokens across various document attributes.")
            
            for attr in attr_cols:
                # We limit unique values to avoid crashing charts with high-cardinality attributes (like IDs)
                unique_count = conn.execute(f"SELECT COUNT(DISTINCT {attr}) FROM corpus").fetchone()[0]
                
                if unique_count > 50:
                    st.warning(f"Attribute **{attr}** has too many unique values ({unique_count}) to visualize effectively.")
                    continue
                    
                attr_data = conn.execute(f"""
                    SELECT 
                        {attr} as Value, 
                        COUNT(*) as Tokens,
                        CAST(COUNT(DISTINCT _token_low) AS FLOAT) / COUNT(*) as TTR
                    FROM corpus 
                    WHERE {attr} IS NOT NULL 
                    GROUP BY {attr} 
                    ORDER BY Tokens DESC
                """).fetch_df()
                
                if not attr_data.empty:
                    st.write(f"**Attribute: {attr}**")
                    ac1, ac2 = st.columns([1, 1])
                    with ac1:
                         fig_a = px.pie(attr_data, names='Value', values='Tokens', title=f"Distribution by {attr}")
                         st.plotly_chart(fig_a, use_container_width=True)
                    with ac2:
                         st.dataframe(
                             attr_data.style.format({'TTR': '{:.4f}'}), 
                             use_container_width=True, 
                             hide_index=True
                         )
                    st.markdown("---")
        else:
            st.caption("No additional XML attributes detected.")

    except Exception as e:
        st.error(f"Error calculating stats: {e}")
    finally:
        conn.close()

def render_online_builder_ui():
    import re
    mode = get_state('online_builder_mode', 'YouTube')
    st.subheader(f"🌐 Online Corpus Builder: {mode}")
    
    if mode == "YouTube":
        st.info("💡 **Experimental:** Max 100,000 words limit for this session.")
        url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        opt = st.radio("Content to Download", ["Transcript only", "Comments only", "Both Transcript and Comments"], index=2)
        
        mode_map = {"Transcript only": "transcript", "Comments only": "comments", "Both Transcript and Comments": "both"}
        
        if st.button("Download YouTube Data", type="primary"):
            if not url:
                st.error("Please enter a URL")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(p)
                    status.caption(m)
                
                with st.spinner("Downloading..."):
                    files, warn = build_online_corpus("youtube", {"url": url, "mode": mode_map[opt]}, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Downloaded {len(files)} components!")
                        if warn: st.warning(warn)
                        st.info("👉 Now click **'Process Downloaded Files'** in the sidebar to index your corpus.")
                    else:
                        st.error(warn or "Failed to download. Ensure the video has a transcript and comments.")

    elif mode == "Link Collection":
        st.info("💡 **Experimental:** Max 50 links and 100,000 words limit.")
        st.caption("Paste one URL per line.")
        links_text = st.text_area("URLs", height=200, placeholder="https://example.com\nhttps://test.org")
        
        if st.button("Scrape Links", type="primary"):
            links = [l.strip() for l in links_text.split('\n') if l.strip()]
            if not links:
                st.error("No links provided")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(p)
                    status.caption(m)
                
                with st.spinner("Scraping..."):
                    files, warn = build_online_corpus("links", {"links": links}, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Scraped {len(files)} pages!")
                        if warn: st.warning(warn)
                        st.info("👉 Now click **'Process Downloaded Files'** in the sidebar to index your corpus.")
                    else:
                        st.error(warn or "Failed to scrape.")

    elif mode == "Keyword Search":
        st.info("💡 **Experimental:** Max 5 keywords and 100,000 words limit.")
        st.caption("Find pages containing at least **n-2** of your keywords (minimum 2).")
        kw_input = st.text_input("Keywords (comma separated)", placeholder="detik, celeb, jokes, kisruh, gosip")
        
        if st.button("Search and Scrape", type="primary"):
            keywords = [k.strip() for k in kw_input.split(',') if k.strip()]
            if not keywords:
                st.error("No keywords provided")
            elif len(keywords) > 5:
                st.error("Max 5 keywords allowed for this experimental feature.")
            else:
                from core.preprocessing.online_corpus import build_online_corpus
                progress_bar = st.progress(0)
                status = st.empty()
                def up(p, m):
                    progress_bar.progress(min(p, 1.0))
                    status.caption(m)
                
                with st.spinner("Searching and scraping..."):
                    files, warn = build_online_corpus("keyword", {"keywords": keywords}, progress_callback=up)
                    if files:
                        set_state('downloaded_online_files', files)
                        st.success(f"✅ Built corpus with {len(files)} matching pages!")
                        if warn: st.warning(warn)
                        st.info("👉 Now click **'Process Downloaded Files'** in the sidebar to index your corpus.")
                    else:
                        st.error(warn or "No matching pages found or search limit exceeded.")

def _render_metadata_annotation_tab(db_path, key_suffix):
    import duckdb
    st.subheader("Metadata Annotation")
    
    meta_tabs = st.tabs(["📄 File Level", "✂️ Segmental Level"])
    
    with meta_tabs[0]:
        st.info("Assign attributes (e.g. Year, Genre, Author) to individual files. These attributes can then be used in **KWIC Restricted Search** and **Sub-corpus Stats**.")
        
        files = ov.get_corpus_files(db_path)
        if not files:
            st.warning("No files found in corpus.")
        else:
            # Get current metadata columns
            conn = duckdb.connect(db_path, read_only=True)
            cols_info = conn.execute("PRAGMA table_info(corpus)").fetch_df()
            conn.close()
            
            standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'topic', 'sentiment'}
            meta_cols = [c for c in cols_info['name'].tolist() if c.lower() not in standard]
            
            # State key for the working dataframe
            state_key = f"meta_editor_df_{db_path}_{key_suffix}"
            
            if get_state(state_key) is None:
                conn = duckdb.connect(db_path, read_only=True)
                # Fetch one sample value per filename for each metadata column
                if meta_cols:
                    select_cols = ", ".join([f"MAX({c}) as {c}" for c in meta_cols])
                    query = f"SELECT filename, {select_cols} FROM corpus GROUP BY filename ORDER BY filename"
                else:
                    query = "SELECT DISTINCT filename FROM corpus ORDER BY filename"
                    
                try:
                    df = conn.execute(query).fetch_df()
                except:
                    df = pd.DataFrame({'filename': files})
                conn.close()
                
                # Ensure all files are represented
                missing = [f for f in files if f not in df['filename'].values]
                if missing:
                    missing_df = pd.DataFrame({'filename': missing})
                    df = pd.concat([df, missing_df], ignore_index=True)
                
                set_state(state_key, df)
            
            df = get_state(state_key)
            
            # 1. Add New Column UI
            c_add1, c_add2 = st.columns([3, 1])
            with c_add1:
                new_col_name = st.text_input("New Attribute Name (e.g. 'Genre')", key=f"new_col_input_{key_suffix}")
            with c_add2:
                st.write(" ") # alignment
                if st.button("➕ Add Attribute", key=f"add_col_btn_{key_suffix}", use_container_width=True):
                    if new_col_name and new_col_name not in df.columns:
                        df[new_col_name] = ""
                        set_state(state_key, df)
                        st.rerun()
                    elif not new_col_name:
                        st.error("Enter a name")
                    else:
                        st.warning("Already exists")

            # 2. Data Editor
            st.write("**Edit File Metadata:**")
            edited_df = st.data_editor(
                df, 
                key=f"meta_editor_widget_{key_suffix}", 
                use_container_width=True, 
                hide_index=True,
                disabled=["filename"]
            )
            
            # 3. Save Button
            if st.button("💾 Apply Metadata Annotation", type="primary", use_container_width=True, key=f"save_meta_btn_{key_suffix}"):
                with st.spinner("Applying to database..."):
                    if ov.apply_metadata_to_files(db_path, edited_df):
                        st.success("✅ Metadata successfully applied to the database!")
                        set_state(state_key, None) # Clear state to force refresh
                        st.info("The new attributes are now available for KWIC filtering and stats.")
                        st.rerun()
                    else:
                        st.error("Failed to apply metadata.")

    with meta_tabs[1]:
        st.info("Annotate specific segments within a file. You can select individual words or whole sentences.")
        
        all_files = ov.get_corpus_files(db_path)
        if not all_files:
            st.warning("No files found.")
            return

        selected_file = st.selectbox("Select File for Segmental Annotation", all_files, key=f"seg_file_select_{key_suffix}")
        
        if selected_file:
            # 1. Word Count Check
            word_count = ov.get_file_word_count(db_path, selected_file)
            st.write(f"**File Size:** {word_count:,} words")
            
            if word_count > 5000:
                st.warning(f"⚠️ This file is too large for segmental annotation ({word_count} > 5000 words).")
                if st.button(f"✂️ Slice '{selected_file}' into 5000-word segments", key=f"slice_btn_{key_suffix}"):
                    with st.spinner("Slicing file..."):
                        if ov.slice_corpus_file(db_path, selected_file, max_words=5000):
                            st.success("File sliced successfully! Please select one of the parts.")
                            st.rerun()
                        else:
                            st.error("Failed to slice file.")
                return

            # 2. Load Tokens and Metadata
            tokens_df = ov.get_file_tokens(db_path, selected_file)
            if tokens_df.empty:
                st.info("No tokens found in this file.")
                return

            # Identify metadata columns
            standard = {'id', 'token', 'pos', 'lemma', 'sent_id', '_token_low', 'filename', 'topic', 'sentiment'}
            meta_cols = [c for c in tokens_df.columns if c.lower() not in standard]
            
            # Helper to check if a row has metadata
            def has_meta(row):
                if not meta_cols: return False
                return any(pd.notna(row[c]) and str(row[c]).strip() != "" for c in meta_cols)

            # --- SELECTION MODE ---
            sel_mode = st.radio("Selection Mode:", ["Word Selection (Natural Grid)", "Sentence Selection (List)"], horizontal=True, key=f"sel_mode_{key_suffix}")
            
            selected_token_ids = []

            if "Word Selection" in sel_mode:
                st.markdown("#### 🖱️ Word Selection Grid")
                st.caption("💡 **How to select:** Click and drag to block a range. Hold **Ctrl** to select multiple separate segments. Words with 🏷️ already have annotations.")
                
                # Prepare grid data with icons for annotated words
                words_per_row = 10
                display_tokens = []
                for _, row in tokens_df.iterrows():
                    token_text = str(row['token'])
                    if has_meta(row):
                        token_text = "🏷️ " + token_text
                    display_tokens.append(token_text)
                
                # Create 2D array
                grid_data = []
                for i in range(0, len(display_tokens), words_per_row):
                    chunk = display_tokens[i:i + words_per_row]
                    if len(chunk) < words_per_row:
                        chunk += [""] * (words_per_row - len(chunk))
                    grid_data.append(chunk)
                
                grid_df = pd.DataFrame(grid_data)
                
                selection_event = st.dataframe(
                    grid_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={i: st.column_config.TextColumn(label="", width="small") for i in range(words_per_row)},
                    on_select="rerun",
                    selection_mode="multi-cell",
                    key=f"token_grid_{selected_file}_{key_suffix}"
                )
                
                selected_cells = selection_event.get("selection", {}).get("cells", [])
                if selected_cells:
                    for cell in selected_cells:
                        r, c = (int(cell["row"]), int(cell["column"])) if isinstance(cell, dict) else (int(cell[0]), int(cell[1]))
                        token_idx = r * words_per_row + c
                        if token_idx < len(tokens_df):
                            selected_token_ids.append(int(tokens_df['id'].iloc[token_idx]))
            
            else:
                st.markdown("#### 📑 Sentence Selection List")
                st.caption("Select one or more sentences to annotate them entirely.")
                
                # Group tokens by sentence
                sentences = []
                for sid, group in tokens_df.groupby('sent_id'):
                    sent_text = " ".join(group['token'].astype(str).tolist())
                    sent_has_meta = group.apply(has_meta, axis=1).any()
                    sentences.append({
                        "sent_id": sid,
                        "Status": "🏷️ Annotated" if sent_has_meta else "Empty",
                        "Text": sent_text,
                        "_ids": group['id'].tolist()
                    })
                
                sent_df = pd.DataFrame(sentences)
                
                # Display sentence list with row selection
                sent_selection = st.dataframe(
                    sent_df[["Status", "Text"]],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    key=f"sent_list_{selected_file}_{key_suffix}"
                )
                
                selected_rows = sent_selection.get("selection", {}).get("rows", [])
                if selected_rows:
                    for r_idx in selected_rows:
                        selected_token_ids.extend(sentences[r_idx]["_ids"])

            # --- ANNOTATION FORM ---
            if selected_token_ids:
                selected_token_ids = sorted(list(set(selected_token_ids)))
                # Preview selection
                sel_mask = tokens_df['id'].isin(selected_token_ids)
                selected_text = " ".join(tokens_df[sel_mask]['token'].astype(str).tolist())
                
                if len(selected_text) > 300:
                    selected_text = selected_text[:300] + "..."
                
                st.success(f"📌 **Selected Segment ({len(selected_token_ids)} tokens):** {selected_text}")

                with st.container(border=True):
                    st.write("**Annotate Selection**")
                    
                    history_key = f"seg_meta_history_{db_path}"
                    if history_key not in st.session_state:
                        st.session_state[history_key] = {"attributes": [], "values": {}}
                    hist = st.session_state[history_key]
                    
                    attr_val_state_key = f"cur_seg_attr_{key_suffix}"
                    val_val_state_key = f"cur_seg_val_{key_suffix}"
                    if attr_val_state_key not in st.session_state: st.session_state[attr_val_state_key] = ""
                    if val_val_state_key not in st.session_state: st.session_state[val_val_state_key] = ""
                    
                    col_attr, col_val = st.columns(2)
                    
                    with col_attr:
                        if hist["attributes"]:
                            st.caption("Reuse attribute:")
                            attr_cols = st.columns(min(len(hist["attributes"]), 4))
                            for i, a in enumerate(hist["attributes"][:4]):
                                if attr_cols[i].button(a, key=f"reuse_attr_{a}_{key_suffix}", use_container_width=True):
                                    st.session_state[attr_val_state_key] = a
                                    st.rerun()
                        
                        attr_input = st.text_input("Attribute (e.g. 'Speaker')", value=st.session_state[attr_val_state_key], key=f"seg_attr_input_{key_suffix}")
                        st.session_state[attr_val_state_key] = attr_input
                    
                    with col_val:
                        if attr_input in hist["values"] and hist["values"][attr_input]:
                            st.caption(f"Reuse for '{attr_input}':")
                            v_list = hist["values"][attr_input]
                            val_cols = st.columns(min(len(v_list), 4))
                            for i, v in enumerate(v_list[:4]):
                                if val_cols[i].button(v, key=f"reuse_val_{v}_{key_suffix}", use_container_width=True):
                                    st.session_state[val_val_state_key] = v
                                    st.rerun()
                        
                        val_input = st.text_input("Value (e.g. 'John')", value=st.session_state[val_val_state_key], key=f"seg_val_input_{key_suffix}")
                        st.session_state[val_val_state_key] = val_input

                    if st.button("💾 Apply Metadata to Selection", type="primary", use_container_width=True, key=f"apply_seg_btn_{key_suffix}"):
                        if not attr_input or not val_input:
                            st.error("Please provide both attribute and value.")
                        else:
                            with st.spinner("Applying..."):
                                meta_dict = {attr_input: val_input}
                                if ov.apply_token_metadata(db_path, selected_token_ids, meta_dict):
                                    st.toast("Metadata applied!", icon="✅")
                                    # Update history
                                    if attr_input not in hist["attributes"]: hist["attributes"].insert(0, attr_input)
                                    if attr_input not in hist["values"]: hist["values"][attr_input] = []
                                    if val_input not in hist["values"][attr_input]: hist["values"][attr_input].insert(0, val_input)
                                    st.session_state[history_key] = hist
                                    st.rerun()
                                else:
                                    st.error("Failed to apply metadata.")
            else:
                st.info("👆 Use the selection tool above to highlight words or sentences for annotation.")

            # --- CURRENT ANNOTATIONS SUMMARY ---
            st.divider()
            st.markdown("#### 📜 Current Segmental Annotations")
            
            if not meta_cols:
                st.info("No segmental metadata has been encoded for this file yet.")
            else:
                # Group tokens into segments with same metadata
                segments = []
                current_seg = None
                for _, row in tokens_df.iterrows():
                    row_meta = {c: row[c] for c in meta_cols if pd.notna(row[c]) and str(row[c]).strip() != ""}
                    if not row_meta:
                        current_seg = None
                        continue
                    if current_seg and current_seg['meta'] == row_meta:
                        current_seg['tokens'].append(row['token'])
                        current_seg['end_id'] = row['id']
                    else:
                        current_seg = {'start_id': row['id'], 'end_id': row['id'], 'tokens': [row['token']], 'meta': row_meta}
                        segments.append(current_seg)
                
                if segments:
                    summary_data = []
                    for seg in segments:
                        for attr, val in seg['meta'].items():
                            summary_data.append({"Range": f"{seg['start_id']}-{seg['end_id']}", "Text": " ".join(seg['tokens']), "Attribute": attr, "Value": val})
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                    
                    with st.expander("🛠️ Advanced: Edit Individual Tokens"):
                        mask = tokens_df[meta_cols].notna().any(axis=1) | (tokens_df[meta_cols] != "").any(axis=1)
                        editable_tokens = tokens_df[mask].copy()
                        if not editable_tokens.empty:
                            edited_tokens = st.data_editor(editable_tokens[['id', 'token'] + meta_cols], key=f"token_editor_{selected_file}_{key_suffix}", hide_index=True, disabled=['id', 'token'], use_container_width=True)
                            if st.button("💾 Save Token Edits", key=f"save_token_edits_{selected_file}"):
                                with st.spinner("Saving..."):
                                    success = True
                                    for col in meta_cols:
                                        val_groups = edited_tokens.groupby(col)
                                        for val, group in val_groups:
                                            ids = group['id'].tolist()
                                            if not ov.apply_token_metadata(db_path, ids, {col: val}): success = False
                                    if success:
                                        st.toast("Edits saved!", icon="✅")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save.")
                else:
                    st.info("No segmental metadata found for this file.")

def _render_reading_ease_tab(db_path, key_suffix=""):
    import duckdb
    import pandas as pd
    
    st.subheader("📖 Reading Ease Analysis")
    
    # 1. Language Warning
    curr_lang = ov.get_corpus_language(db_path)
    if curr_lang and curr_lang.lower() not in ['en', 'english']:
        st.warning("⚠️ **Non-English Language Warning:** Readability formulas are designed and calibrated for English. For other languages, they serve as structural estimations (based on word, sentence, character, and syllable ratios) but do not strictly correspond to standard English school grades.")
    
    # 2. Clickable Transparency Link/Popover
    st.markdown("For full transparency on how readability metrics are calculated and categorized:")
    with st.expander("🔍 Click here to view mathematical formulas and difficulty classification mapping", expanded=False):
        st.markdown("""
        ### Readability Metrics & Classification Transparency
        
        #### 1. Flesch-Kincaid Grade Level
        Calculates U.S. school grade level difficulty.
        * **Formula**: `0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59`
        * **Interpretation**: Represents the educational grade level required to understand the text (e.g. 6 = 6th grade, 12 = high school senior, 16+ = university level).
        
        #### 2. Gunning Fog Index
        Measures text complexity based on sentence length and complex words.
        * **Formula**: `0.4 * ((words / sentences) + 100 * (complex_words / words))`
        * *Complex words* are defined as words containing 3 or more syllables.
        * **Interpretation**: Under 8 is easy, 8–12 is standard, 12–16 is difficult, and 17+ is very difficult.
        
        #### 3. Coleman-Liau Index
        Measures readability based on character counts and sentence ratios instead of syllables.
        * **Formula**: `0.0588 * L - 0.296 * S - 15.8`
        * *L* = average number of letters per 100 words.
        * *S* = average number of sentences per 100 words.
        * **Interpretation**: Standard grade level output (e.g. 6 = 6th grade).
        
        #### 4. Automated Readability Index (ARI)
        Calculates grade level based on characters per word and words per sentence.
        * **Formula**: `4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43`
        * **Interpretation**: Standard grade level output.
        
        #### 5. SMOG Grade
        Predicts comprehension based on the count of polysyllabic words.
        * **Formula**: `1.0430 * sqrt(complex_words * 30 / sentences) + 3.1291`
        * **Interpretation**: Standard grade level output (e.g. 10 = 10th grade).
        
        ---
        
        ### Unified Difficulty Classification Matrix
        We calculate all 5 formulas for each sentence, average them, and assign the sentence to one of these 5 discrete brackets:
        
        | Bracket Name | Numerical Range (Average Grade Level) | Education / Comprehension Level |
        | :--- | :--- | :--- |
        | **1. Very Easy** | 6.0 or less | Elementary School level (up to Grade 6) |
        | **2. Easy** | from 6.1 to 8.0 | Junior High / Middle School level (Grades 6–8) |
        | **3. Standard** | from 8.1 to 12.0 | High School level (Grades 8–12) |
        | **4. Difficult** | from 12.1 to 16.0 | College / University Prep level (Grades 12–16) |
        | **5. Very Difficult** | greater than 16.0 | Graduate / Professional level (Grades 16+) |
        """)
        
    st.markdown("---")
    
    # 3. Perform Calculations
    with st.spinner("Analyzing readability metrics..."):
        sentence_df = get_sentence_stats(db_path)
        
    if sentence_df.empty:
        st.info("No valid text data found in the corpus.")
        return
        
    metrics_data = compute_readability_metrics(sentence_df)
    
    # 4. Display Overall metrics
    st.markdown("#### 📊 Overall Corpus Readability")
    overall = metrics_data['overall']['metrics']
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Flesch-Kincaid", f"{overall['Flesch-Kincaid Grade Level']}")
    m2.metric("Gunning Fog", f"{overall['Gunning Fog']}")
    m3.metric("Coleman-Liau", f"{overall['Coleman-Liau']}")
    m4.metric("ARI", f"{overall['ARI']}")
    m5.metric("SMOG", f"{overall['SMOG']}")
    
    # Combined average and bracket
    avg_score = round(sum(overall.values()) / len(overall), 2)
    overall_bracket = map_score_to_level(avg_score)
    st.info(f"**Overall Corpus Classification:** {overall_bracket} (Average Grade Level: **{avg_score}**)")
    
    # 5. Display File-level table
    st.markdown("#### 📄 Corpus Files Readability")
    file_rows = []
    for fname, fmetrics in metrics_data['files'].items():
        favg = round(sum(fmetrics.values()) / len(fmetrics), 2)
        fbracket = map_score_to_level(favg)
        file_rows.append({
            'Filename': fname,
            'Flesch-Kincaid': fmetrics['Flesch-Kincaid Grade Level'],
            'Gunning Fog': fmetrics['Gunning Fog'],
            'Coleman-Liau': fmetrics['Coleman-Liau'],
            'ARI': fmetrics['ARI'],
            'SMOG': fmetrics['SMOG'],
            'Average GL': favg,
            'Difficulty Level': fbracket
        })
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    
    # 5.1. Chunk-Level Readability Breakdown
    st.markdown("#### ✂️ Chunk-Level Readability Breakdown")
    st.caption("Analyze the text in sequential blocks of words. This is useful for single-file corpora to see readability progression and identify difficult passages.")
    
    col_ch1, col_ch2 = st.columns([1, 2])
    with col_ch1:
        selected_chunk_size = st.selectbox(
            "Select Chunk Size (Words):",
            options=[100, 1000, 10000, 100000],
            index=1, # Default to 1000
            key=f"chunk_size_select_{key_suffix}"
        )
        
    with st.spinner(f"Analyzing readability per {selected_chunk_size:,} words..."):
        chunk_stats = get_chunk_readability_stats(db_path, selected_chunk_size)
        
    if chunk_stats:
        chunk_df = pd.DataFrame(chunk_stats)
        st.dataframe(chunk_df, use_container_width=True, hide_index=True)
    else:
        st.info("No chunk statistics generated.")
        
    # 6. Display Sub-corpora grouping
    st.markdown("#### 🧱 Sub-corpora Readability")
    sub_options = list(metrics_data['subcorpora'].keys())
    
    if not sub_options:
        st.info("No sub-corpora attributes (such as Topic, Sentiment, or XML Attributes) detected.")
    else:
        selected_sub = st.selectbox(
            "Select Sub-corpus Grouping Category:",
            options=sub_options,
            key=f"sub_readability_select_{key_suffix}"
        )
        if selected_sub:
            sub_rows = []
            for gval, smetrics in metrics_data['subcorpora'][selected_sub].items():
                savg = round(sum(smetrics.values()) / len(smetrics), 2)
                sbracket = map_score_to_level(savg)
                sub_rows.append({
                    selected_sub: gval,
                    'Flesch-Kincaid': smetrics['Flesch-Kincaid Grade Level'],
                    'Gunning Fog': smetrics['Gunning Fog'],
                    'Coleman-Liau': smetrics['Coleman-Liau'],
                    'ARI': smetrics['ARI'],
                    'SMOG': smetrics['SMOG'],
                    'Average GL': savg,
                    'Difficulty Level': sbracket
                })
            st.dataframe(pd.DataFrame(sub_rows), use_container_width=True, hide_index=True)
            
    # 7. Database Annotation Section
    st.divider()
    st.markdown("#### 🚀 Database Readability Annotation")
    st.caption("Annotate the corpus database with Reading Ease difficulty levels. Once annotated, the levels will appear as sub-corpora, and you can restrict searches to specific difficulty ranges using the filter panels.")
    
    conn = duckdb.connect(db_path, read_only=True)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(corpus)").fetchall()]
    conn.close()
    
    has_reading_ease_level = 'reading_ease_level' in cols
    if has_reading_ease_level:
        st.success("✅ **Reading Ease levels are already annotated in this corpus.** You can re-run annotation at any time if the corpus text changes.")
    else:
        st.info("Reading Ease levels have not been annotated yet. Run annotation below to enable difficulty-level filtering.")
    
    # Selection of annotation unit / scope
    col_ann1, col_ann2 = st.columns([1, 1])
    with col_ann1:
        ann_scope = st.radio(
            "**Annotation Granularity:**",
            options=["Sentence Level", f"Chunks (per {selected_chunk_size:,} words)"],
            help="Sentence Level: Calculates difficulty for every individual sentence (great for multi-document/varied corpora). Chunk Level: Breaks the text into segments of the selected size (recommended for single large texts/flat corpora to see difficulty segments).",
            key=f"ann_scope_{key_suffix}"
        )
        
    if st.button("Annotate Reading Ease Levels", key=f"btn_annotate_reading_ease_{key_suffix}", type="primary"):
        if ann_scope == "Sentence Level":
            with st.spinner("Analyzing and annotating sentences..."):
                filenames = []
                sent_ids = []
                levels = []
                
                for _, row in sentence_df.iterrows():
                    smetrics = calculate_formulas(
                        int(row['words']),
                        int(row['sentences']),
                        int(row['syllables']),
                        int(row['characters']),
                        int(row['complex_words'])
                    )
                    savg = sum(smetrics.values()) / len(smetrics)
                    slevel = map_score_to_level(savg)
                    
                    filenames.append(row['filename'])
                    sent_ids.append(row['sent_id'])
                    levels.append(slevel)
                    
                if apply_reading_ease_annotation(db_path, filenames, sent_ids, levels):
                    st.toast("Reading Ease Levels Annotated successfully!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to write annotations to database.")
        else:
            with st.spinner(f"Analyzing and annotating {selected_chunk_size:,}-word chunks..."):
                if annotate_reading_ease_by_chunks(db_path, chunk_size=selected_chunk_size):
                    st.toast(f"Reading Ease Levels Annotated by {selected_chunk_size:,}-word chunks successfully!", icon="✅")
                    st.rerun()
                else:
                    st.error("Failed to write annotations to database.")

