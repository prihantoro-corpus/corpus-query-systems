import streamlit as st
import os
import shutil
from ui_streamlit.state_manager import set_state, get_state, reset_tool_states
from ui_streamlit.utils import notify_timing
from core.preprocessing.corpus_loader import load_monolingual_corpus_files, load_built_in_corpus
from core.modules.overview import calculate_corpus_statistics
from core.config import get_available_corpora, BUILT_IN_CORPUS_DETAILS, STANZA_LANG_MAP

def render_sidebar():
    """
    Renders the sidebar for corpus selection and settings.
    Returns: The selected view name.
    """
    # Active Corpus Display Banner in Sidebar
    current_path = get_state('current_corpus_path')
    active_corpus_name = get_state('current_corpus_name')
    if current_path:
        display_name = "USER CORPUS" if active_corpus_name == "Uploaded Batch" else active_corpus_name
        
        comp_path = get_state('comp_corpus_path')
        comp_name = get_state('comp_corpus_name')
        if get_state('comparison_mode') and comp_path:
            display_comp = "USER CORPUS" if comp_name == "Uploaded Batch" else comp_name
            st.sidebar.markdown(f"<div style='background-color:#1e293b; padding:10px; border-radius:8px; border:1px solid #00ADB5; margin-bottom:15px;'>📂 <span style='color:#00FFF5; font-weight:bold;'>Active:</span> <span style='color:white; font-weight:bold;'>{display_name}</span> vs <span style='color:white; font-weight:bold;'>{display_comp}</span></div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"<div style='background-color:#1e293b; padding:10px; border-radius:8px; border:1px solid #00ADB5; margin-bottom:15px;'>📂 <span style='color:#00FFF5; font-weight:bold;'>Active:</span> <span style='color:white; font-weight:bold;'>{display_name}</span></div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div style='background-color:#1e293b; padding:10px; border-radius:8px; border:1px solid #475569; margin-bottom:15px;'>📂 <span style='color:#94a3b8; font-weight:bold;'>Active:</span> <span style='color:#94a3b8; font-weight:bold;'>None Loaded</span></div>", unsafe_allow_html=True)

    # 1. Navigation (Tools) - MOVED TO TOP
    st.sidebar.title("Tools (v1.1 Stanza)")
    st.sidebar.caption("App Version: v300526")
    view = st.sidebar.radio(
        "Go to", 
        ["Overview", "Concordance", "N-Gram", "Collocation", "Word Profiler", "Dictionary", "Keyword", "Distribution", "Statistical Testing", "Summarisation", "Quiz Creation"]
    )
    
    st.sidebar.markdown("---")
    
    # 2. Corpus Selection
    st.sidebar.title("Corpus Selection")
    
    # Corpus Type Selection
    corpus_type = st.sidebar.radio(
        "Corpus Type", 
        ["Monolingual", "Parallel"],
        index=0 if get_state('corpus_type') == "Monolingual" else 1
    )
    
    if corpus_type != get_state('corpus_type'):
        set_state('corpus_type', corpus_type)
        reset_tool_states()
        st.rerun()

    # Comparison Mode Toggle
    comparison_mode = st.sidebar.checkbox("Enable Comparison Mode", value=get_state('comparison_mode', False))
    if comparison_mode != get_state('comparison_mode'):
        set_state('comparison_mode', comparison_mode)
        st.rerun()
        
    # Corpus Source
    if 'sidebar_source_selectbox' not in st.session_state:
        st.session_state['sidebar_source_selectbox'] = "Upload Files"

    source_type = st.sidebar.radio(
        "Source", 
        ["Upload Files", "Built-in Corpora", "Online Corpus"],
        key="sidebar_source_selectbox"
    )
    
    # Update backward compatible state backend
    set_state('source_type', source_type)
    
    if source_type == "Online Corpus":
        online_mode = st.sidebar.radio("Builder Mode", ["YouTube", "Link Collection", "Keyword Search"])
        set_state('online_builder_mode', online_mode)
    
    current_path = get_state('current_corpus_path')


    # 3. Current Status Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Corpus")
    if current_path:
        display_name = "USER CORPUS" if get_state('current_corpus_name') == "Uploaded Batch" else get_state('current_corpus_name')
        st.sidebar.success(f"Primary: **{display_name}**")
        
    comp_path = get_state('comp_corpus_path')
    if get_state('comparison_mode') and comp_path:
        display_comp = "USER CORPUS" if get_state('comp_corpus_name') == "Uploaded Batch" else get_state('comp_corpus_name')
        st.sidebar.info(f"Comparison: **{display_comp}**")
    elif get_state('comparison_mode'):
        st.sidebar.warning("Load a 2nd corpus to compare.")
    
    if not current_path and not comp_path:
        st.sidebar.warning("No Corpus Loaded")
        
    st.sidebar.markdown("---")
    
    # 4. AI Interpretation Settings
    st.sidebar.title("AI Interpretation")
    
    # AI Provider Selection
    ai_provider = st.sidebar.radio("AI Provider", ["Ollama", "Gemini"], 
                                   index=0 if get_state('ai_provider') == "Ollama" else 1,
                                   key="sidebar_ai_provider")
    set_state('ai_provider', ai_provider)

    if ai_provider == "Gemini":
        gemini_key = st.sidebar.text_input("Gemini API Key", value=get_state('gemini_api_key', ''), type="password", key="sidebar_gemini_key")
        set_state('gemini_api_key', gemini_key)
        st.sidebar.caption("Google Gemini 1.5 Flash (Cloud fallback)")
    else:
        # Connection Check Button (Always Visible)
        if st.sidebar.button("Check Local AI Status"):
            from core.ai_service import test_ollama_connection
            current_url = get_state('ollama_url')
            success, msg = test_ollama_connection(current_url)
            if success: st.sidebar.success(msg)
            else: st.sidebar.error(msg)
                
        with st.sidebar.expander("Local AI Settings", expanded=False):
            o_url = st.text_input("Ollama URL", value=get_state('ollama_url'), key="sidebar_ollama_url")
            from core.ai_service import get_available_models
            
            # Cache model fetching to avoid network lag on every rerun
            @st.cache_data(ttl=60, show_spinner=False)
            def get_cached_models(url):
                return get_available_models(url)
            
            available_models = get_cached_models(o_url)
            current_model = get_state('ai_model')
            if available_models:
                if current_model not in available_models: available_models.append(current_model)
                index = available_models.index(current_model) if current_model in available_models else 0
                o_model = st.radio("Ollama Model", available_models, index=index, horizontal=True, key="sidebar_ai_model_select")
            else:
                o_model = st.text_input("Model (Manual)", value=current_model, key="sidebar_ai_model")
            
            if o_url != get_state('ollama_url'): set_state('ollama_url', o_url)
            if o_model != get_state('ai_model'): set_state('ai_model', o_model)

            # Initialize install states
            if 'ollama_install_step' not in st.session_state:
                st.session_state['ollama_install_step'] = None
            if 'ollama_download_thread' not in st.session_state:
                st.session_state['ollama_download_thread'] = None

            # Force reload the installer utility module to prevent Streamlit module caching issues
            import sys
            import importlib
            if 'core.utils.installer' in sys.modules:
                try:
                    importlib.reload(sys.modules['core.utils.installer'])
                except Exception:
                    pass

            st.markdown("---")
            st.markdown("**Local AI Installer**")

            # Show installation status messages if any
            if 'ollama_install_message' in st.session_state and st.session_state['ollama_install_message']:
                msg_type, msg_text = st.session_state['ollama_install_message']
                if msg_type == "info":
                    st.info(msg_text)
                elif msg_type == "error":
                    st.error(msg_text)
                st.session_state['ollama_install_message'] = None

            # Step 1: Initial state
            if st.session_state['ollama_install_step'] is None:
                if st.button("Get Ollama (One-Click)", key="sidebar_install_ollama_btn"):
                    st.session_state['ollama_install_step'] = "confirm"
                    st.rerun()

            # Step 2: Confirmation state
            elif st.session_state['ollama_install_step'] == "confirm":
                st.warning("This will consume 3 GB of your hard drive (including models). Continue?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Continue", key="confirm_yes"):
                        from core.utils.installer import get_ollama_download_url, OllamaDownloadThread, is_ollama_installed
                        import tempfile
                        import platform
                        
                        if is_ollama_installed():
                            st.session_state['ollama_install_message'] = ("info", "Ollama is already installed on your system!")
                            st.session_state['ollama_install_step'] = None
                            st.rerun()
                        
                        url, ext = get_ollama_download_url()
                        if url:
                            temp_dir = tempfile.gettempdir()
                            dest_path = os.path.join(temp_dir, f"OllamaSetup{ext}")
                            
                            # Start background thread
                            thread = OllamaDownloadThread(url, dest_path)
                            thread.start()
                            
                            st.session_state['ollama_download_thread'] = thread
                            st.session_state['ollama_install_step'] = "downloading"
                        else:
                            st.error(f"Unsupported platform: {platform.system()}. Please install manually from https://ollama.com")
                            st.session_state['ollama_install_step'] = None
                        st.rerun()
                with col2:
                    if st.button("No, Cancel", key="confirm_no"):
                        st.session_state['ollama_install_step'] = None
                        st.rerun()

            # Step 3: Downloading state
            elif st.session_state['ollama_install_step'] == "downloading":
                thread = st.session_state['ollama_download_thread']
                if thread is not None:
                    # Show progress bar and status
                    progress_bar = st.progress(thread.progress)
                    st.caption(thread.status)
                    
                    # Show Cancel button
                    if st.button("Cancel Download", key="cancel_download_btn"):
                        thread.cancelled = True
                        st.session_state['ollama_install_step'] = "cancelled"
                        st.rerun()
                    
                    # If thread finished
                    if not thread.is_alive():
                        if thread.completed:
                            from core.utils.installer import run_ollama_installer
                            success, run_err = run_ollama_installer(thread.dest_path)
                            if success:
                                st.success("Installer launched!")
                                if run_err:
                                    st.info(run_err)
                                else:
                                    st.info("Please follow the setup wizard. Once installed, start the Ollama application and click 'Check Local AI Status' above.")
                            else:
                                st.error(run_err)
                            st.session_state['ollama_install_step'] = None
                            st.session_state['ollama_download_thread'] = None
                        elif thread.error:
                            st.error(f"Download failed: {thread.error}")
                            st.session_state['ollama_install_step'] = None
                            st.session_state['ollama_download_thread'] = None
                        st.rerun()
                    else:
                        # Rerun to update progress bar
                        import time
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.session_state['ollama_install_step'] = None
                    st.rerun()

            # Step 4: Cancelled cleanup state
            elif st.session_state['ollama_install_step'] == "cancelled":
                st.info("Download cancelled. Cleaned up temporary files.")
                st.session_state['ollama_install_step'] = None
                st.session_state['ollama_download_thread'] = None
                st.rerun()



    st.sidebar.markdown("---")
    
    # 5. CORTEX Assistant (App Usage Chat)
    st.sidebar.title("🧠 CORTEX Assistant")
    st.sidebar.caption("Ask how to use the app or about corpus linguistics.")
    
    chat_hist = get_state('sidebar_chat_history', [])
    with st.sidebar.container(height=250):
        for msg in chat_hist:
            with st.chat_message("user" if "user" in msg else "assistant"):
                st.markdown(msg["content"])
    
    if prompt := st.sidebar.chat_input("How do I...?", key="sidebar_chat_input"):
        chat_hist.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            from core.ai_service import app_guide_chat
            response, err = app_guide_chat(
                user_query=prompt, 
                chat_history=[{"user": m["content"], "ai": chat_hist[i+1]["content"]} for i, m in enumerate(chat_hist[:-1]) if m["role"] == "user"],
                api_key=get_state('gemini_api_key') if get_state('ai_provider') == "Gemini" else None
            )
            if response:
                chat_hist.append({"role": "assistant", "content": response})
            else:
                chat_hist.append({"role": "assistant", "content": f"Sorry, I encountered an error: {err}"})
        set_state('sidebar_chat_history', chat_hist)
        st.rerun()

    return view
