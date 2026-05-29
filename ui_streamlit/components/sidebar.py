import streamlit as st
import os
import shutil
from ui_streamlit.state_manager import set_state, get_state, reset_tool_states
from core.preprocessing.corpus_loader import load_monolingual_corpus_files, load_built_in_corpus
from core.modules.overview import calculate_corpus_statistics
from core.config import get_available_corpora, BUILT_IN_CORPUS_DETAILS, STANZA_LANG_MAP

def render_sidebar():
    """
    Renders the sidebar for corpus selection and settings.
    Returns: The selected view name.
    """
    # 1. Navigation (Tools) - MOVED TO TOP
    st.sidebar.title("Tools (v1.1 Stanza)")
    st.sidebar.caption("App Version: v290526")
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
    source_options = ["Upload Files", "Built-in Corpora", "Online Corpus"]
    saved_source = get_state('source_type', "Upload Files")
    try:
        source_index = source_options.index(saved_source)
    except ValueError:
        source_index = 0

    source_type = st.sidebar.selectbox(
        "Source", 
        source_options,
        index=source_index,
        key="sidebar_source_selectbox"
    )
    
    if source_type != saved_source:
        set_state('source_type', source_type)
        st.rerun()
    
    if source_type == "Online Corpus":
        online_mode = st.sidebar.radio("Builder Mode", ["YouTube", "Link Collection", "Keyword Search"])
        set_state('online_builder_mode', online_mode)
    
    current_path = get_state('current_corpus_path')
    
    if source_type in ("Upload Files", "Online Corpus"):
        if source_type == "Upload Files":
            uploaded_files = st.sidebar.file_uploader(
                "Upload Corpus Files (XML, TXT, CSV, XLSX)", 
                accept_multiple_files=True,
                type=['xml', 'txt', 'csv', 'xlsx']
            )
        else:
            uploaded_files = [] # No manual uploads in online mode
        
        # Language and Format Selection
        lang_col, fmt_col = st.sidebar.columns(2)
        with lang_col:
            # Prepare language list. Add 'OTHER' at the end.
            lang_options = list(STANZA_LANG_MAP.keys()) + ["OTHER"]
            selected_lang_label = st.selectbox(
                "Language", 
                lang_options, 
                index=0,
                key="upload_language_select"
            )
            
            # Map label to code for processing
            if selected_lang_label == "OTHER":
                lang_code = "OTHER"
            else:
                lang_code = STANZA_LANG_MAP[selected_lang_label]
                
        with fmt_col:
            fmt = st.selectbox(
                "Format", 
                ["Raw (Natural text)", "Tagged (Vertical)"], 
                index=0,
                key="upload_format_select"
            )
        
        btn_label = "Process Downloaded Files" if source_type == "Online Corpus" else "Process Uploaded Files"
        files_to_process = uploaded_files
        
        if source_type == "Online Corpus":
            downloaded = get_state('downloaded_online_files', [])
            if not downloaded:
                st.sidebar.warning("No files downloaded yet. Use the Online Corpus Builder in the main area.")
            else:
                st.sidebar.info(f"Ready to process {len(downloaded)} downloaded files.")
                # Convert dicts to file-like objects
                import io
                files_to_process = []
                for f_dict in downloaded:
                    buf = io.BytesIO(f_dict['content'].encode('utf-8'))
                    buf.name = f_dict['filename']
                    files_to_process.append(buf)

        if files_to_process:
            if st.sidebar.button(btn_label):
                # Force reload logic to pick up hotfixes
                import sys
                import importlib
                import core.preprocessing.corpus_loader as cl
                try:
                    for mod in ['core.preprocessing.tagging', 'core.preprocessing.xml_parser', 'core.preprocessing.corpus_loader']:
                        if mod in sys.modules:
                            importlib.reload(sys.modules[mod])
                    st.toast("Processing modules updated! 🔄")
                except Exception as e:
                    print(f"Reload Error: {e}")

                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.caption(text)

                with st.spinner("Processing Corpus..."):
                    result = cl.load_monolingual_corpus_files(
                        files_to_process, 
                        explicit_lang_code=lang_code,
                        selected_format=fmt,
                        progress_callback=update_progress
                    )
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        if result.get('warning'):
                            st.warning(result['warning'])
                            
                        if not get_state('comparison_mode'):
                            set_state('current_corpus_path', result['db_path'])
                            set_state('corpus_stats', result['stats'])
                            set_state('current_corpus_name', "Uploaded Batch")
                            set_state('xml_structure_data', result.get('structure'))
                            set_state('target_lang', lang_code)
                        else:
                            if not get_state('current_corpus_path'):
                                set_state('current_corpus_path', result['db_path'])
                                set_state('corpus_stats', result['stats'])
                                set_state('current_corpus_name', "Primary")
                                set_state('xml_structure_data', result.get('structure'))
                            else:
                                set_state('comp_corpus_path', result['db_path'])
                                set_state('comp_corpus_stats', result['stats'])
                                set_state('comp_corpus_name', "Comparison")
                                set_state('comp_xml_structure_data', result.get('structure'))
                        
                        st.success("Corpus Loaded Successfully!")
                        st.rerun()

    elif source_type == "Built-in Corpora":
        built_in_corpora = get_available_corpora()
        
        if not built_in_corpora:
            st.sidebar.warning("No corpora found in local 'corpora' directory.")
            selected_names = []
        else:
            selected_corpus = st.sidebar.selectbox(
                "Select Corpus",
                options=["Select a corpus..."] + list(built_in_corpora.keys()),
                index=0,
                key="builtin_selected_corpus"
            )
            selected_names = []
            if selected_corpus and selected_corpus != "Select a corpus...":
                selected_names = [selected_corpus]
            
            # Show info for first selected corpus
            if selected_names:
                detail = BUILT_IN_CORPUS_DETAILS.get(selected_names[0])
                if detail:
                    with st.sidebar.expander("ℹ️ Corpus Info"):
                        st.markdown(detail, unsafe_allow_html=True)

            if st.sidebar.button("Load Built-in", disabled=not selected_names):
                # Force reload of parser logic to pick up hotfixes
                import sys
                import importlib
                try:
                    # Reload parser and loader
                    for mod in ['core.preprocessing.xml_parser', 'core.preprocessing.corpus_loader']:
                        if mod in sys.modules:
                            importlib.reload(sys.modules[mod])
                    
                    # Reload analytical modules to pick up hotfixes
                    for mod in ['core.modules.concordance', 'core.modules.collocation', 'core.modules.distribution', 'core.modules.statistical_testing', 'ui_streamlit.caching']:
                        if mod in sys.modules:
                            importlib.reload(sys.modules[mod])
                    
                    # Clear Streamlit cache to force re-execution of queries
                    st.cache_data.clear()
                    
                    # Re-import the function from the reloaded module
                    from core.preprocessing.corpus_loader import load_built_in_corpus
                    st.toast("Internal search modules reloaded! 🚀")
                except Exception as e:
                    st.sidebar.error(f"Reload Error: {e}")
                    print(f"Reload Error: {e}")

                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.caption(text)

                with st.spinner("Downloading and processing..."):
                    # Get URLs for all selected corpora
                    selected_urls = [built_in_corpora[name] for name in selected_names]
                    
                    result = load_built_in_corpus(
                        selected_names, 
                        selected_urls,
                        progress_callback=update_progress
                    )
                    
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        # Create combined name
                        combined_name = " + ".join(selected_names)
                        
                        if not get_state('comparison_mode'):
                            set_state('current_corpus_path', result['db_path'])
                            set_state('corpus_stats', result['stats'])
                            set_state('current_corpus_name', combined_name)
                            set_state('xml_structure_data', result.get('structure'))
                            
                            # Language is already saved in DB by load_built_in_corpus
                        else:
                            if not get_state('current_corpus_path'):
                                set_state('current_corpus_path', result['db_path'])
                                set_state('corpus_stats', result['stats'])
                                set_state('current_corpus_name', combined_name)
                                set_state('xml_structure_data', result.get('structure'))
                                # Language is already saved in DB by load_built_in_corpus
                            else:
                                set_state('comp_corpus_path', result['db_path'])
                                set_state('comp_corpus_stats', result['stats'])
                                set_state('comp_corpus_name', combined_name)
                                set_state('comp_xml_structure_data', result.get('structure'))
                        
                        st.success("Built-in Corpus Loaded!")
                        st.rerun()


    # 3. Current Status Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Corpus")
    if current_path:
        st.sidebar.success(f"Primary: **{get_state('current_corpus_name')}**")
        
    comp_path = get_state('comp_corpus_path')
    if get_state('comparison_mode') and comp_path:
        st.sidebar.info(f"Comparison: **{get_state('comp_corpus_name')}**")
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
                o_model = st.selectbox("Ollama Model", available_models, index=index, key="sidebar_ai_model_select")
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
