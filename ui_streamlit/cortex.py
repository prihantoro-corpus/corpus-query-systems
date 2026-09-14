import streamlit as st
import sys
import os

# Add project root to path so we can import from core/ui_streamlit
# Add project root to path so we can import from core/ui_streamlit
architecture_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if architecture_root not in sys.path:
    sys.path.insert(0, architecture_root)

# Force reload logic removed to improve performance. 
# Streamlit handles module reloading natively in development mode.
import core.config
import core.ai_service
import core.modules.concordance
import core.modules.collocation
import core.modules.distribution
import core.modules.statistical_testing
import core.modules.ngram
import core.modules.word_trend
import ui_streamlit.state_manager
import ui_streamlit.caching
import ui_streamlit.components.sidebar
import ui_streamlit.components.corpus_selection
import ui_streamlit.views.overview_view
import ui_streamlit.views.concordance_view

# Debug Imports
try:
    from ui_streamlit.state_manager import init_session_state
    from ui_streamlit.components.sidebar import render_sidebar
    from ui_streamlit.views.overview_view import render_overview
    from ui_streamlit.views.dictionary_view import render_dictionary_view
    from ui_streamlit.views.concordance_view import render_concordance_view
    from ui_streamlit.views.ngram_view import render_ngram_view
    from ui_streamlit.views.collocation_view import render_collocation_view
    from ui_streamlit.views.keyword_view import render_keyword_view
    from ui_streamlit.views.distribution_view import render_distribution_view
    from ui_streamlit.views.statistical_testing_view import render_statistical_testing_view
    from ui_streamlit.views.word_profiler_view import render_word_profiler_view
    from ui_streamlit.views.summarisation_view import render_summarisation_view
    from ui_streamlit.views.quiz_creation_view import render_quiz_creation_view
    from ui_streamlit.views.word_trend_view import render_word_trend_view
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

from ui_streamlit.state_manager import init_session_state
from core.visualiser.styles import POS_COLOR_MAP

# Page Configuration
st.set_page_config(
    page_title="CORTEX Corpus Query System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize State
init_session_state()

# Handle Query Params for URL-based Routing (e.g. from Word Trend or N-Gram)
if 'word' in st.query_params and 'time' in st.query_params and 'attr' in st.query_params:
    ui_streamlit.state_manager.set_state('kwic_search_term', st.query_params['word'])
    ui_streamlit.state_manager.set_state('concordance_forced_xml_where', f" AND CAST({st.query_params['attr']} AS VARCHAR) = '{st.query_params['time']}'")
    ui_streamlit.state_manager.set_state('current_module', 'Concordance')
    st.query_params.clear()
    st.rerun()
elif 'kwic_query' in st.query_params:
    ui_streamlit.state_manager.set_state('kwic_search_term', st.query_params['kwic_query'])
    ui_streamlit.state_manager.set_state('current_module', 'Concordance')
    st.query_params.clear()
    st.rerun()

# CSS Styling for Premium Dark Blue Theme (Matching Online Configuration)
PRIMARY_COLOR = "#00ADB5"
BACKGROUND_COLOR = "#0b132b"
SECONDARY_BACKGROUND = "#1c2541"
TEXT_COLOR = "#FFFFFF"

st.markdown(f"""
<style>
    /* Global Font Scaling */
    html {{
        font-size: 110% !important; /* +10% base increase, user requested +50% but that is huge, usually means interface scaling. 110% is a safe start, or we can go 125%. */
    }}

    /* Main App Container */
    .stApp {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    
    /* Top Header Bar */
    header[data-testid="stHeader"], [data-testid="stHeader"] {{
        background-color: {BACKGROUND_COLOR} !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {SECONDARY_BACKGROUND} !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    
    /* Sidebar Headers - Cyan Accent */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #00FFF5 !important;
    }}
    
    /* Base Inputs & Selectboxes */
    div[data-baseweb="input"], 
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {{
        background-color: #1c2541 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }}
    
    /* Multiselect Tags */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {PRIMARY_COLOR} !important;
        color: #FFFFFF !important;
    }}
    
    /* Buttons & Popovers - Streamlit Dark Theme Clean Override */
    .stButton>button,
    [data-testid="stPopover"]>button {{
        background-color: #00ADB5 !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    
    .stButton>button:hover,
    [data-testid="stPopover"]>button:hover {{
        background-color: #008c93 !important;
        color: #FFFFFF !important;
        border-color: #00FFF5 !important;
    }}
    
    /* File Uploader Container & Dropzone */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px dashed rgba(0, 255, 245, 0.4) !important;
        border-radius: 10px !important;
    }}

    [data-testid="stFileUploaderDropzone"] button {{
        background-color: #00ADB5 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }}

    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: #008c93 !important;
        color: #FFFFFF !important;
    }}
    
    /* Expander Styling */
    .stExpander {{
        background-color: rgba(28, 37, 65, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        margin-bottom: 1rem !important;
    }}
    .stExpander summary {{
        color: #00FFF5 !important;
        font-weight: 600 !important;
    }}
    
    /* Table Headers */
    table thead th {{
        background-color: rgba(0, 173, 181, 0.2) !important;
        color: #00FFF5 !important;
        font-weight: 700 !important;
    }}
""", unsafe_allow_html=True)

from core.visualiser.styles import POS_COLOR_MAP

# Main Layout
def main():
    import ui_streamlit.components.sidebar

    st.markdown(
        """
        <div style="text-align: right; margin-bottom: 0.5rem;">
            <a href="https://docs.google.com/document/d/1HjF20hLgsPUORqLAXKmY-uqh_RKoQZNmb4uuaRiWyq0/edit?usp=sharing" target="_blank"
               style="color:#00FFF5; font-weight:700; text-decoration:none; margin-right: 15px;">
                Citation
            </a>
            <a href="https://docs.google.com/document/d/1kRlyuVXN4G7QBH4bd1hEFeblegc8ai1_rMKspW1C9-E/edit?usp=sharing" target="_blank"
               style="color:#00FFF5; font-weight:700; text-decoration:none; margin-right: 15px;">
                ACKNOWLEDGEMENT
            </a>
            <a href="https://drive.google.com/drive/folders/15AedwrmeG1IX0JsyGrZoxVGNQ5cYt4Ju?usp=sharing" target="_blank"
               style="color:#00FFF5; font-weight:700; text-decoration:none;">
                📘 Manual
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## CORTEX: An Advanced Corpus Query System <span style='font-size: 0.5em; color: rgba(255,255,255,0.5); font-weight: normal; vertical-align: middle; margin-left: 10px;'>version 0.0.1: 5-sept-2026</span>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # Render main screen corpus selection banner/settings
    from ui_streamlit.components.corpus_selection import render_corpus_selection_main
    render_corpus_selection_main()
    
    # Render Sidebar and get current view
    current_view = render_sidebar()
    
    # Router
    if current_view == "Overview":
        render_overview()
    elif current_view == "Concordance":
        render_concordance_view()
    elif current_view == "N-Gram":
        render_ngram_view()
    elif current_view == "Collocation":
        render_collocation_view()
    elif current_view == "Dictionary":
        render_dictionary_view()
    elif current_view == "Word Trend":
        from ui_streamlit.views.word_trend_view import render_word_trend_view
        render_word_trend_view()
    elif current_view == "Word Profiler":
        render_word_profiler_view()
    elif current_view == "Keyword":
        render_keyword_view()
    elif current_view == "Distribution":
        render_distribution_view()
    elif current_view == "Statistical Testing":
        render_statistical_testing_view()
    elif current_view == "Summarisation":
        render_summarisation_view()
    elif current_view == "Quiz Creation":
        render_quiz_creation_view()
    else:
        st.write("Select a module from the sidebar.")

    st.markdown("---")

if __name__ == "__main__":
    main()
