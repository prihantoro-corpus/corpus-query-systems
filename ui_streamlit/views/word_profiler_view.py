import streamlit as st
import pandas as pd
import os
import altair as alt
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.utils import notify_timing
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions, get_xml_attribute_columns
from core.modules.word_profiler import load_wordlist, run_word_profiler_analysis
from core.io_utils import df_to_excel_bytes

def render_word_profiler_view():
    st.header("Word Profiler")
    
    corpus_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Corpus')
    
    if not corpus_path:
        st.warning("Please load a corpus first.")
        return

    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Word Profiler")

    with col_main:


        st.markdown("Analyze your corpus coverage using one or more wordlists.")

        # 1. Configuration
        with st.expander("Analysis Settings", expanded=True):
            basis = st.radio("Analysis Basis", ["Whole Corpus", "By Filename", "By Metadata"], horizontal=True, key="wp_basis")

            metadata_col = None
            if basis == "By Metadata":
                import duckdb
                con = duckdb.connect(corpus_path, read_only=True)
                attr_cols = get_xml_attribute_columns(con)
                con.close()
                if attr_cols:
                    metadata_col = st.radio("Select Metadata Attribute", attr_cols, horizontal=True, key="wp_metadata_col")
                else:
                    st.warning("No metadata attributes found in this corpus.")
                    metadata_col = None

        # 2. Wordlist Selection
        with st.expander("Wordlist Selection", expanded=True):
            wordlist_source = st.radio("Source", ["Existing Wordlist", "Upload Your Own"], horizontal=True, key="wp_wl_source")

            selected_wordlists = {} # name: wordlist_dict
            if wordlist_source == "Existing Wordlist":
                # List files in wordlist directory
                wl_dir = "wordlist"
                available_lists = []
                for root, dirs, files in os.walk(wl_dir):
                    for file in files:
                        if file.endswith(".txt"):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, wl_dir)
                            available_lists.append(rel_path)

                if available_lists:
                    chosen_lists = st.multiselect("Choose Wordlist(s)", sorted(available_lists), key="wp_wl_chosen")
                    for chosen in chosen_lists:
                        selected_wordlists[chosen] = load_wordlist(os.path.join(wl_dir, chosen))
                else:
                    st.info("No wordlists found in the `wordlist/` directory.")
            else:
                uploaded_files = st.file_uploader("Upload Wordlist(s) (Plain or Categorised .txt)", type=["txt"], accept_multiple_files=True)
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        content = uploaded_file.read().decode('utf-8')
                        selected_wordlists[uploaded_file.name] = load_wordlist(content, is_file=False)

        # 3. Filtering
        xml_filters = render_xml_restriction_filters(corpus_path, "word_profiler", corpus_name=corpus_name)
        xml_where, xml_params = apply_xml_restrictions(xml_filters)

        # 4. Run Analysis
        if st.button("Run Analysis", type="primary"):
            if not selected_wordlists:
                st.error("Please select or upload at least one wordlist.")
            else:
                with st.spinner("Analyzing..."):
                    all_results = {}
                    for wl_name, wl_dict in selected_wordlists.items():
                        df_res = notify_timing(f"Word Profiler analysis for '{wl_name}' completed")(run_word_profiler_analysis)(
                            db_path=corpus_path,
                            wordlist=wl_dict,
                            basis=basis,
                            metadata_col=metadata_col,
                            xml_where_clause=xml_where,
                            xml_params=xml_params
                        )
                        all_results[wl_name] = df_res
                    set_state('last_wp_results_multiple', all_results)

        # 5. Results
        all_results = get_state('last_wp_results_multiple')
        if all_results is not None:
            if not all_results:
                st.info("No results found.")
            else:
                st.subheader("Analysis Results")
                for wl_name, df_results in all_results.items():
                    with st.expander(f"📊 Results for: {wl_name}", expanded=True):
                        if df_results.empty:
                            st.info(f"No results for {wl_name}")
                            continue

                        st.dataframe(df_results, use_container_width=True)

                        # --- Visualization ---
                        st.markdown("#### 📊 Visualization")
                        render_word_profiler_chart(df_results, wl_name)

                        # Download Button for this specific wordlist
                        st.download_button(
                            label=f"Download {wl_name} Results (Excel)",
                            data=df_to_excel_bytes(df_results),
                            file_name=f"word_profiler_{corpus_name}_{wl_name.split('.')[0]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{wl_name}"
                        )

                        # Summary Metrics
                        if basis == "Whole Corpus" and not df_results.empty:
                            st.markdown("#### Coverage Summary")
                            # Filter out 'Segment' and 'Total Tokens'
                            res_cols = [c for c in df_results.columns if c not in ['Segment', 'Total Tokens']]
                            cols = st.columns(min(len(res_cols) // 2, 6)) # Max 6 metrics per row
                            for i in range(0, len(res_cols), 2):
                                cat_name = str(res_cols[i]).replace(" Freq", "")
                                freq = df_results.iloc[0][res_cols[i]]
                                perc = df_results.iloc[0][res_cols[i+1]]
                                with cols[(i // 2) % len(cols)]:
                                    st.metric(cat_name, f"{freq:,}", f"{perc}%")

    def render_word_profiler_chart(df, wl_name):
        """
        Renders a bar chart for the Word Profiler results using Altair.
        """
        if df.empty:
            return

        # Transform data to long format for Altair
        # Columns: Segment, Cat1 Freq, Cat1 %, ...
        cat_cols = [c for c in df.columns if c.endswith(" %")]

        chart_data = []
        for _, row in df.iterrows():
            segment = row['Segment']
            for col in cat_cols:
                cat_name = col.replace(" %", "")
                percentage = row[col]
                freq_col = col.replace(" %", " Freq")
                frequency = row[freq_col] if freq_col in df.columns else 0
                chart_data.append({
                    'Segment': segment,
                    'Category': cat_name,
                    'Percentage': percentage,
                    'Frequency': frequency
                })

        plot_df = pd.DataFrame(chart_data)

        if len(df) == 1:
            # Whole Corpus - Simple Bar Chart
            chart = alt.Chart(plot_df).mark_bar(color='#00ADB5').encode(
                x=alt.X('Category:N', title='Category', sort=None),
                y=alt.Y('Percentage:Q', title='Percentage (%)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Category', 'Percentage', 'Frequency']
            ).properties(height=300)
        else:
            # Multiple Segments - Stacked Bar Chart
            # Sort Categories to put OOV at the end if possible
            cats = plot_df['Category'].unique().tolist()
            if 'OOV' in cats:
                cats.remove('OOV')
                cats.append('OOV')

            chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X('Segment:N', title='Segment', sort=None),
                y=alt.Y('Percentage:Q', title='Percentage (%)', stack="normalize"), # Or stack=True for raw but normalized is better for coverage
                color=alt.Color('Category:N', sort=cats, scale=alt.Scale(scheme='category20')),
                tooltip=['Segment', 'Category', 'Percentage', 'Frequency']
            ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)


