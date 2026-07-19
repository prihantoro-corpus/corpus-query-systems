import streamlit as st
import pandas as pd
import io
from ui_streamlit.state_manager import get_state, set_state
from core.modules.word_trend import get_available_metadata_attributes, get_metadata_values, get_emerging_words, get_word_tracker_data, compute_tracker_statistics
from core.modules.overview import get_unique_pos_tags
from core.ai_service import interpret_results_llm

def render_results_section(df_display, df_full, selected_attr, key_prefix):
    if df_display is None or df_display.empty:
        st.warning("No words found for this category.")
        return
        
    # Export button
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_full.to_excel(writer, index=False, sheet_name='Words')
    excel_data = excel_buffer.getvalue()
    
    st.download_button(
        label=f"📥 Export Results to Excel",
        data=excel_data,
        file_name=f"{key_prefix}_words.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_{key_prefix}"
    )
    
    # Format for UI display
    grouped_data = []
    if not df_display.empty:
        for time_val, group in df_display.groupby('time_val', sort=False):
            word_parts = []
            for _, row in group.iterrows():
                word = row['token']
                freq_text = f"{row['rel_freq']:.1f} pmw" if 'rel_freq' in row else f"{row.get('freq', 0)} abs"
                link = f'<a href="?word={word}&time={time_val}&attr={selected_attr}" target="_self" style="color:#00FFF5; text-decoration:none;">{word}</a> ({freq_text})'
                word_parts.append(link)
                
            words_str = ", ".join(word_parts)
            grouped_data.append({"Time Period": time_val, "Words (Relative Freq pmw)": words_str})
        
    df_grouped = pd.DataFrame(grouped_data)
    
    st.markdown("### 🔍 Interactive Results")
    st.info("Click on any word to automatically open its concordance lines.")
    
    view_mode = st.radio("Display Mode", ["Tabular View", "Bar Graph View"], horizontal=True, key=f"view_{key_prefix}")
    
    if view_mode == "Tabular View":
        html_table = df_grouped.to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        for time_val, group in df_display.groupby('time_val', sort=False):
            st.markdown(f"#### Period: {time_val}")
            max_freq = group['rel_freq'].max() if 'rel_freq' in group.columns else group.get('freq', 0).max()
            
            chart_data = []
            for _, row in group.iterrows():
                word = row['token']
                freq_val = row['rel_freq'] if 'rel_freq' in row else row.get('freq', 0)
                concordance_url = f"?word={word}&time={time_val}&attr={selected_attr}"
                chart_data.append({
                    "Word": concordance_url,
                    "Relative Frequency (pmw)": freq_val
                })
                
            df_chart = pd.DataFrame(chart_data)
            
            st.dataframe(
                df_chart,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Word": st.column_config.LinkColumn(
                        "Word",
                        help="Click to view in Concordance",
                        display_text=r"\?word=([^&]+)"
                    ),
                    "Relative Frequency (pmw)": st.column_config.ProgressColumn(
                        "Frequency (pmw)",
                        help="Relative Frequency per million words",
                        format="%.1f",
                        min_value=0,
                        max_value=float(max_freq)
                    )
                }
            )
            
    st.markdown("---")
    if st.button("🤖 Interpret Results with AI", key=f"btn_ai_{key_prefix}"):
        with st.spinner("AI is analyzing the words..."):
            description = f"{key_prefix.capitalize()} Words Analysis based on time periods"
            stats_text = df_grouped.to_string(index=False)
            response, error = interpret_results_llm(
                target_word="Multiple Words",
                analysis_type=f"{key_prefix} words trend",
                data_description=description,
                data=stats_text
            )
            if error:
                st.error(error)
            else:
                set_state(f'wt_ai_res_{key_prefix}', response)
                
    ai_res = get_state(f'wt_ai_res_{key_prefix}')
    if ai_res:
        st.markdown("### 🧠 AI Interpretation")
        st.markdown(ai_res)

def render_word_trend_view():
    import importlib
    import core.modules.word_trend
    importlib.reload(core.modules.word_trend)
    global get_available_metadata_attributes, get_metadata_values, get_emerging_words, get_word_tracker_data, compute_tracker_statistics
    get_available_metadata_attributes = core.modules.word_trend.get_available_metadata_attributes
    get_metadata_values = core.modules.word_trend.get_metadata_values
    get_emerging_words = core.modules.word_trend.get_emerging_words
    get_word_tracker_data = core.modules.word_trend.get_word_tracker_data
    compute_tracker_statistics = core.modules.word_trend.compute_tracker_statistics
    
    st.header("Word Trend")
    
    corpus_path = get_state('current_corpus_path')
    if not corpus_path:
        st.warning("Please load a corpus from the Overview module first.")
        return
        
    st.markdown("Analyze how vocabulary changes over time.")
    
    # Render the global tabs at the top
    tab_tracker, tab_excl, tab_emerge = st.tabs(["📈 Word Tracker (Custom)", "💎 Exclusive Words (Unique to Period)", "🌱 Emerging Words (Chronological)"])
    
    with tab_tracker:
        render_word_tracker_tab(corpus_path)

    with tab_excl:
        render_trend_tab(
            corpus_path=corpus_path,
            mode='exclusive',
            title="Exclusive Words (Unique to Period)",
            description="Identify words that **only appear** in one specific period and are completely absent from all other selected periods.",
            key_prefix="exclusive"
        )
        
    with tab_emerge:
        render_trend_tab(
            corpus_path=corpus_path,
            mode='chronological',
            title="Emerging Words (Chronological)",
            description="Identify words that **emerge for the first time** in the corpus over a chronological timeline.",
            key_prefix="emerge"
        )

def render_trend_tab(corpus_path, mode, title, description, key_prefix):
    st.markdown(description)
    
    # 1. Select Metadata Attribute
    st.subheader(f"1. Temporal Metadata Configuration")
    
    attributes = get_available_metadata_attributes(corpus_path)
    if not attributes:
        st.warning("No metadata attributes found in the corpus. Word Trend requires metadata (e.g. time, year, period).")
        return
        
    selected_attr = st.radio("Select Time Metadata Attribute", attributes, horizontal=True, key=f"attr_{key_prefix}")
    
    if selected_attr:
        values = get_metadata_values(corpus_path, selected_attr)
        if not values:
            st.warning(f"No values found for attribute '{selected_attr}'.")
            return
            
        st.markdown("**Period Selection**")
        if mode == 'chronological':
            st.info("The order here matters. Words emerging in a later period must not have appeared in any previous period in this list.")
        else:
            st.info("Select the periods to compare. Words shown will be exclusively found in that specific period and absent from all others selected.")
            
        ordered_values = st.multiselect(
            "Time Periods", 
            options=values, 
            default=values,
            key=f"periods_{key_prefix}"
        )
        
        # 2. POS Filter
        st.subheader("2. Part-of-Speech Filter (Optional)")
        
        available_pos = get_unique_pos_tags(corpus_path)
        
        pos_mode = st.radio("POS Filter Mode", ["Include (Analyze Only)", "Exclude"], horizontal=True, key=f"posmode_{key_prefix}")
        
        st.markdown("**Select POS Tags**")
        selected_pos = []
        if available_pos:
            num_cols = min(len(available_pos), 6)
            cols = st.columns(num_cols) if num_cols > 0 else []
            for i, tag in enumerate(available_pos):
                with cols[i % num_cols]:
                    if st.checkbox(tag, key=f"pos_{key_prefix}_{tag}"):
                        selected_pos.append(tag)
        
        backend_pos_mode = 'include' if 'Include' in pos_mode else 'exclude'
        
        # 3. Output Configuration
        st.subheader("3. Output Settings")
        top_n = st.radio("Number of words to display per period", [10, 20, 30, 40, 50, 100], horizontal=True, key=f"topn_{key_prefix}")
        
        if st.button(f"Apply / Analyze {title}", type="primary", key=f"btn_{key_prefix}"):
            if not ordered_values:
                st.error("Please select at least one time period.")
            else:
                with st.spinner(f"Analyzing {title.lower()}..."):
                    df_display, df_full = get_emerging_words(
                        corpus_path, 
                        selected_attr, 
                        ordered_values, 
                        backend_pos_mode, 
                        selected_pos, 
                        top_n,
                        mode
                    )
                    
                    if df_display.empty:
                        st.warning("No words found or an error occurred.")
                    else:
                        set_state(f'wt_df_display_{key_prefix}', df_display)
                        set_state(f'wt_df_full_{key_prefix}', df_full)
                        set_state(f'wt_attr_{key_prefix}', selected_attr)
        
        # Display Results if available
        df_display = get_state(f'wt_df_display_{key_prefix}')
        df_full = get_state(f'wt_df_full_{key_prefix}')
        
        if df_display is not None and not df_display.empty:
            st.markdown("---")
            st.subheader("Results")
            render_results_section(df_display, df_full, get_state(f'wt_attr_{key_prefix}'), key_prefix)

def render_word_tracker_tab(corpus_path):
    st.markdown("Track the relative frequency (per million words) of specific words across time periods.")
    
    # Temporal Metadata
    st.subheader("1. Temporal Metadata Configuration")
    attributes = get_available_metadata_attributes(corpus_path)
    if not attributes:
        st.warning("No metadata attributes found in the corpus.")
        return
        
    selected_attr = st.radio("Select Time Metadata Attribute", attributes, horizontal=True, key="attr_tracker")
    
    if selected_attr:
        # Word Input
        st.subheader("2. Words to Track")
        tracker_mode = st.radio("Search Mode", ["Simple", "Advanced"], horizontal=True, key="tracker_search_mode")
        
        if tracker_mode == "Simple":
            st.info("Input words separated by a comma. (e.g. technology, computer, internet, smartphone, digital)")
            tracked_words_input = st.text_input("Enter words", key="tracker_words_input_simple")
            words_to_track = [w.strip() for w in tracked_words_input.split(',')] if tracked_words_input else []
            tracker_basis = "Word"
        else:
            st.info("Advanced query. Use POS, tag, lemma, wildcard like you use in advanced concordance. Use the ➕ button to add more queries to compare. (e.g. can_NN or [go]_V*)")
            
            tracker_basis = st.radio("Output Basis", ["Word", "Lemma"], horizontal=True, key="tracker_basis_adv")
            
            if 'tracker_adv_queries' not in st.session_state:
                st.session_state['tracker_adv_queries'] = [""]
                
            for i, q in enumerate(st.session_state['tracker_adv_queries']):
                col_input, col_btn = st.columns([10, 1])
                with col_input:
                    st.session_state['tracker_adv_queries'][i] = st.text_input(f"Query {i+1}", value=q, key=f"tracker_adv_query_{i}", label_visibility="collapsed")
                with col_btn:
                    if i > 0:
                        if st.button("✖", key=f"btn_rem_adv_{i}"):
                            st.session_state['tracker_adv_queries'].pop(i)
                            st.rerun()
                            
            if st.button("➕ Add Query", key="btn_add_adv"):
                st.session_state['tracker_adv_queries'].append("")
                st.rerun()
                
            words_to_track = [q.strip() for q in st.session_state['tracker_adv_queries'] if q.strip()]
        
        st.subheader("3. Inferential Statistics & Interpretation")
        stat_options = {
            "Correlation": "Do words fluctuate together? (Spearman Rank)",
            "Trend Comparison": "Which word grows/shrinks faster? (Linear Regression Slope)",
            "Variance/Volatility": "Which word is most unstable? (Coefficient of Variation & Levene's Test)",
            "Prediction": "Does a spike in one word precede another? (Lagged Correlation)"
        }
        
        captions = [
            "Library: scipy.stats.spearmanr | Formula: ρ = 1 - (6∑dᵢ²) / (n(n²-1))",
            "Library: numpy.polyfit | Formula: y = mx + c",
            "Library: numpy.std, scipy.stats.levene | Formula: CV = σ/μ",
            "Library: scipy.stats.spearmanr (Lagged) | Formula: ρ(X_t, Y_{t+1})"
        ]
        
        selected_stat = st.radio("Select an analysis to run automatically:", list(stat_options.keys()), 
                                 format_func=lambda x: f"{x} - {stat_options[x]}",
                                 captions=captions)
        
        if st.button("Generate Chart & Analysis", type="primary", key="btn_tracker"):
            if not words_to_track:
                st.error("Please enter at least one word or query to track.")
            else:
                is_advanced = (tracker_mode == "Advanced")
                
                with st.spinner("Generating chart and computing statistics..."):
                    df_chart = get_word_tracker_data(corpus_path, selected_attr, words_to_track, is_advanced=is_advanced, basis=tracker_basis)
                    
                    if df_chart.empty:
                        st.warning("None of the tracked words were found in the corpus across the time periods.")
                        set_state('wt_tracker_df', None)
                        set_state('wt_stat_type', None)
                        set_state('wt_stat_result', None)
                    else:
                        set_state('wt_tracker_df', df_chart)
                        set_state('wt_stat_type', selected_stat)
                        set_state('wt_stat_result', None) # Clear previous AI response
                        
        # Display Chart
        df_chart = get_state('wt_tracker_df')
        if df_chart is not None and not df_chart.empty:
            st.markdown("---")
            st.subheader("Word Tracking Over Time")
            
            show_trendline = st.checkbox("Show Linear Regression Trendline")
            import numpy as np
            
            # Prepare plotting dataframe
            plot_df = df_chart.copy()
            
            if show_trendline and len(plot_df) > 1:
                x = np.arange(len(plot_df))
                for word in plot_df.columns:
                    y = plot_df[word].values
                    m, b = np.polyfit(x, y, 1)
                    trend_y = m * x + b
                    trend_y = np.maximum(trend_y, 0) # Clamp to 0
                    plot_df[word] = trend_y
            
            # 1. Aggregate Chart (All Words)
            st.markdown("#### 📊 Aggregate View (All Words)")
            st.line_chart(plot_df, use_container_width=True)
            
            st.markdown("---")
            
            # 2. Individual Charts
            st.info("Showing individual trends (Relative Frequency per million words) for each tracked word.")
            for word in plot_df.columns:
                st.markdown(f"#### 📈 {word.capitalize()}")
                word_df = plot_df[[word]].copy()
                # Rename the column to avoid Altair/Vega-Lite parsing errors with bracketed column names like '[see]'
                word_df.columns = ["Relative Frequency"]
                st.line_chart(word_df, use_container_width=True)
                
            # 3. Statistical Interpretation (Rubric Based)
            stat_type = get_state('wt_stat_type')
            if stat_type:
                st.markdown("---")
                
                with st.spinner("Computing statistics..."):
                    stat_data = compute_tracker_statistics(df_chart, stat_type)
                    
                if "error" in stat_data:
                    st.error(stat_data["error"])
                else:
                    st.subheader(f"📊 {stat_data['title']}")
                    if "subtitle" in stat_data and stat_data["subtitle"]:
                        st.info(stat_data["subtitle"])
                        
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("**Results**")
                        st.dataframe(stat_data["results_df"], use_container_width=True, hide_index=True)
                        
                    with col2:
                        st.markdown("**Reference Rubric**")
                        st.dataframe(stat_data["rubric_df"], use_container_width=True, hide_index=True)
                        
                    st.markdown("---")
                    if st.button("🤖 Interpret Statistical Results with AI", key="btn_tracker_ai"):
                        with st.spinner("AI is analyzing the statistical results..."):
                            description = f"Word Tracker ({stat_type}) analysis for words: {', '.join(df_chart.columns)}"
                            stats_text = stat_data["results_df"].to_string()
                            response, error = interpret_results_llm(
                                target_word="Multiple Words",
                                analysis_type=stat_type,
                                data_description=description,
                                data=stats_text
                            )
                            if error:
                                st.error(error)
                            else:
                                set_state('wt_tracker_ai_result', response)
                                
                    ai_res = get_state('wt_tracker_ai_result')
                    if ai_res:
                        st.markdown("### 🧠 AI Interpretation")
                        st.markdown(ai_res)
