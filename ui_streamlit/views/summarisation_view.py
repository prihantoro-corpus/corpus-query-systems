import streamlit as st
import pandas as pd
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.utils import notify_timing
from core.modules.summarisation import (
    get_summarization_metadata_fields, 
    get_metadata_values, 
    get_text_data, 
    summarize_text_extractive,
    summarize_text_ai
)

def render_summarisation_view():
    st.title("📄 Text Summarisation")
    st.markdown("""
        Generate concise summaries of your corpus or specific segments. 
        Choose between **Traditional Extractive** methods or **AI-powered** summarization.
    """)

    db_path = get_state('current_corpus_path')
    if not db_path:
        st.warning("Please load a corpus in the sidebar first.")
        return

    # Layout: Settings on the left, Results on the right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Summarisation Settings")
        
        basis = st.radio(
            "Basis for Summarisation", 
            ["Overall Corpus", "By Metadata", "By File Name"],
            index=0,
            horizontal=True
        )
        
        field = None
        selected_values = []
        
        if basis == "By Metadata":
            fields = get_summarization_metadata_fields(db_path)
            if fields:
                field = st.radio("Select Metadata Field", fields, horizontal=True)
                values = get_metadata_values(db_path, field)
                selected_values = st.multiselect("Select Value(s)", values, default=[values[0]] if values else [])
            else:
                st.info("No metadata fields found in this corpus.")
        
        elif basis == "By File Name":
            fields = get_summarization_metadata_fields(db_path)
            filename_fields = [f for f in fields if 'file' in f.lower() or 'doc' in f.lower()]
            if filename_fields:
                field = st.radio("Select File Field", filename_fields, horizontal=True)
                values = get_metadata_values(db_path, field)
                selected_values = st.multiselect("Select File(s)", values, default=[values[0]] if values else [])
            else:
                st.warning("Could not automatically identify a 'File Name' field. Please use 'By Metadata' and select the appropriate field.")
        else:
            selected_values = [None] # Overall mode

        word_limit = st.slider("Target Word Count", min_value=10, max_value=500, value=100, step=10)
        
        method = st.radio("Summarisation Method", ["Traditional (Extractive)", "AI Powered"])
        
        algorithm = "Luhn"
        if method == "Traditional (Extractive)":
            algorithm = st.radio("Algorithm", ["Luhn", "LexRank", "Lsa", "KL"], horizontal=True)
            st.caption("Extractive summarization picks the most important sentences from the original text.")
        else:
            provider = get_state('ai_provider', 'Ollama')
            model = get_state('ai_model', 'phi3:latest')
            st.info(f"Using {provider} ({model})")
            st.caption("AI summarization generates new text based on the original content.")

        if st.button("Generate Summary", type="primary", use_container_width=True):
            if not selected_values:
                st.error("Please select at least one value to summarize.")
            else:
                with st.spinner(f"Processing {len(selected_values)} segment(s)..."):
                    summaries_result = {}
                    
                    # Common parameters
                    lang_code = get_state('target_lang', 'english')
                    lang_map = {"en": "english", "id": "indonesian", "ja": "japanese", "zh": "chinese"}
                    full_lang = lang_map.get(lang_code, "english")
                    
                    for val in selected_values:
                        display_name = val if val else "Overall Corpus"
                        text = get_text_data(db_path, basis, field, val)
                        
                        if not text:
                            summaries_result[display_name] = "Error: No text found for this segment."
                            continue
                        
                        if method == "Traditional (Extractive)":
                            summary = notify_timing(f"Extractive summary of {display_name} generated")(summarize_text_extractive)(text, full_lang, word_limit, algorithm)
                        else:
                            summary = notify_timing(f"AI summary of {display_name} generated")(summarize_text_ai)(
                                text, 
                                provider=get_state('ai_provider'),
                                model=get_state('ai_model'),
                                api_key=get_state('gemini_api_key'),
                                word_limit=word_limit
                            )
                        summaries_result[display_name] = summary
                    
                    set_state('summaries_dict', summaries_result)
                    set_state('last_basis', basis)

    with col2:
        summaries = get_state('summaries_dict', {})
        basis_label = get_state('last_basis', "")
        
        if summaries:
            st.subheader(f"Summaries: {basis_label}")
            
            # Use tabs for multiple summaries
            if len(summaries) > 1:
                tabs = st.tabs(list(summaries.keys()))
                for i, (name, content) in enumerate(summaries.items()):
                    with tabs[i]:
                        st.markdown(f"""
                            <div style="background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(0, 255, 245, 0.2); line-height: 1.6;">
                                {content}
                            </div>
                        """, unsafe_allow_html=True)
                        st.download_button(f"Download {name}", data=content, file_name=f"summary_{name}.txt", mime="text/plain", key=f"dl_{i}")
            else:
                # Single summary display
                name, content = list(summaries.items())[0]
                st.info(f"**Source:** {name}")
                st.markdown(f"""
                    <div style="background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(0, 255, 245, 0.2); line-height: 1.6;">
                        {content}
                    </div>
                """, unsafe_allow_html=True)
                st.download_button("Download Summary", data=content, file_name="summary.txt", mime="text/plain")

            st.markdown("---")
            if st.button("Clear Results"):
                set_state('summaries_dict', {})
                st.rerun()
        else:
            st.subheader("Generated Summary")
            st.write("Configure the settings on the left and click 'Generate Summary' to see results.")
            
            # Premium Visual: Blank state or illustration
            st.markdown(
                """
                <div style="text-align: center; color: #666; margin-top: 50px;">
                    <span style="font-size: 100px;">📚</span>
                    <p>Select multiple aspects to see batch summaries here.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
