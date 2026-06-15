import streamlit as st
import datetime
import importlib
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions

import core.modules.quiz_creation
importlib.reload(core.modules.quiz_creation)
import core.modules.diffit_generator
importlib.reload(core.modules.diffit_generator)
from core.modules.quiz_creation import (
    generate_full_quiz,
    get_corpus_sentences,
    generate_section_a,
    generate_section_b,
    generate_section_c,
    generate_section_d,
    create_exercises_docx,
    create_answer_key_docx,
    rebuild_sentence_text
)

def render_quiz_creation_view():
    # Helper callbacks for mutually exclusive algorithm checkboxes
    def select_luhn():
        if st.session_state.get("diffit_algo_luhn"):
            st.session_state.diffit_algo_lexrank = False
            st.session_state.diffit_algo_lsa = False
            st.session_state.diffit_algo_kl = False

    def select_lexrank():
        if st.session_state.get("diffit_algo_lexrank"):
            st.session_state.diffit_algo_luhn = False
            st.session_state.diffit_algo_lsa = False
            st.session_state.diffit_algo_kl = False

    def select_lsa():
        if st.session_state.get("diffit_algo_lsa"):
            st.session_state.diffit_algo_luhn = False
            st.session_state.diffit_algo_lexrank = False
            st.session_state.diffit_algo_kl = False

    def select_kl():
        if st.session_state.get("diffit_algo_kl"):
            st.session_state.diffit_algo_luhn = False
            st.session_state.diffit_algo_lexrank = False
            st.session_state.diffit_algo_lsa = False

    st.markdown("""
        <style>
        @keyframes pulse-blank {
            0% {
                box-shadow: 0 0 4px rgba(0, 255, 245, 0.2);
                border-color: rgba(0, 255, 245, 0.6);
            }
            50% {
                box-shadow: 0 0 12px rgba(0, 255, 245, 0.6);
                border-color: rgba(0, 255, 245, 1);
            }
            100% {
                box-shadow: 0 0 4px rgba(0, 255, 245, 0.2);
                border-color: rgba(0, 255, 245, 0.6);
            }
        }
        .premium-blank-box {
            display: inline-block;
            min-width: 95px;
            height: 18px;
            vertical-align: middle;
            background: rgba(0, 173, 181, 0.08);
            border: 1.5px dashed #00FFF5;
            border-radius: 6px;
            margin: 0 6px;
            box-shadow: 0 0 6px rgba(0, 255, 245, 0.15);
            animation: pulse-blank 2.5s infinite ease-in-out;
            transition: all 0.3s ease;
        }
        .premium-blank-box:hover {
            background: rgba(0, 255, 245, 0.15);
            border-style: solid;
            transform: scale(1.03);
            box-shadow: 0 0 15px rgba(0, 255, 245, 0.8);
        }
        </style>
        <div style="background-color: rgba(0, 173, 181, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #00ADB5; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #00FFF5; font-size: 1.8rem;">🧠 Quiz Creation</h2>
            <p style="margin: 5px 0 0 0; color: #FFFFFF; font-size: 1rem;">
                Automatically generate corpus-driven language exercises from your uploaded corpora. 
                Perfect for teachers, material designers, and independent learners.
            </p>
        </div>
    """, unsafe_allow_html=True)

    db_path = get_state('current_corpus_path')
    corpus_name = get_state('current_corpus_name', 'Unnamed Corpus')
    
    if not db_path:
        st.warning("⚠️ Please select and load a corpus in the sidebar first to generate quizzes.")
        st.markdown("""
            <div style="text-align: center; margin-top: 50px; opacity: 0.7;">
                <span style="font-size: 80px;">📂</span>
                <h4 style="color: #FFFFFF;">No Active Corpus Source</h4>
                <p style="color: #A0AEC0;">Use the "Corpus Selection" panel in the sidebar to upload files or choose a built-in corpus.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    # Guidelines Layout using shared component
    from ui_streamlit.components.guidelines import render_guidelines
    col_main = render_guidelines("Quiz Creation")

    with col_main:


        # Two main tabs: Automatic Mode & Setting Mode
        tab_auto, tab_setting = st.tabs(["⚡ Automatic Mode", "⚙️ Setting Mode"])

        # =====================================================================
        # TAB: SETTING MODE (UNDER CONSTRUCTION)
        # =====================================================================
        with tab_setting:
            st.subheader("⚙️ Differentiated Resource Generator (Setting Mode)")
            
            # --- AI Connection Check Warning ---
            provider = get_state('ai_provider', 'Ollama')
            ai_connected = False
            ai_message = ""
            
            if provider == "Gemini":
                connected = get_state('gemini_connected', False)
                api_key = get_state('gemini_api_key', '')
                if connected and api_key:
                    ai_connected = True
                    ai_message = f"🟢 Connected to Google Gemini (Model: {get_state('gemini_model', 'gemini-2.5-flash')})"
                else:
                    ai_connected = False
                    ai_message = "🔴 Gemini API is not connected. Please enter your API key and connect in the sidebar."
            else: # Ollama
                url = get_state('ollama_url', 'http://127.0.0.1:11434/api/generate')
                model = get_state('ai_model', 'phi3:latest')
                from core.ai_service import test_ollama_connection
                success, msg = test_ollama_connection(url)
                if success:
                    ai_connected = True
                    ai_message = f"🟢 Connected to local Ollama (Model: {model})"
                else:
                    ai_connected = False
                    ai_message = f"🔴 Ollama server is not reachable at {url}. Ensure Ollama is running."

            if not ai_connected:
                st.warning(f"⚠️ **AI connection warning:** {ai_message}\nSetting Mode supports AI-based abstractive summarisation, but you can still use traditional methods offline.")
            else:
                st.success(ai_message)

            col_cfg, col_preview = st.columns([1.2, 2])

            with col_cfg:
                st.markdown("### 🛠️ Step 1: Text Extraction Setup")
                
                is_manual_select = st.session_state.get("diffit_extraction_method") == "Select Text Parts"
                
                # Word Count N
                target_words = st.number_input(
                    "Target Word Count (N)",
                    min_value=10,
                    max_value=2000,
                    value=500,
                    step=50,
                    key="diffit_target_words",
                    disabled=is_manual_select
                )
                
                # Extraction Method
                extraction_method = st.radio(
                    "Extraction Method",
                    ["Automatically Summarise", "Chunk Text As Is", "Select Text Parts"],
                    key="diffit_extraction_method"
                )
                
                # Select corpus document
                from core.modules.summarisation import get_metadata_values, get_text_data
                files = get_metadata_values(db_path, 'filename') or []
                
                st.write("**Select Document(s) from Corpus:**")
                random_checked = st.checkbox("🎲 Random Document", value=False, key="diffit_random_doc_chk")
                
                selected_files = []
                if files:
                    with st.expander("📄 Choose specific files", expanded=not random_checked):
                        for f in files:
                            chk_key = f"diffit_chk_{f}"
                            f_checked = st.checkbox(f, value=False, disabled=random_checked, key=chk_key)
                            if f_checked and not random_checked:
                                selected_files.append(f)
                else:
                    st.info("No documents found in corpus.")
                
                # Load selected documents' text dynamically for manual range selection
                full_raw_text = ""
                total_words_count = 0
                if not random_checked and selected_files:
                    texts = []
                    for doc in selected_files:
                        txt = get_text_data(db_path, "By File Name", "filename", doc)
                        if txt:
                            texts.append(txt)
                    full_raw_text = "\n\n".join(texts)
                    total_words_count = len(full_raw_text.split())
                
                # Manual Range Selection UI
                chosen_start = 1
                chosen_end = 100
                if extraction_method == "Select Text Parts":
                    if random_checked:
                        st.warning("⚠️ Manual text selection is not compatible with 'Random Document'. Please uncheck 'Random Document' and select specific files above.")
                    elif not selected_files:
                        st.info("📄 Please select at least one document from the list above to choose text parts.")
                    else:
                        st.write(f"**Document Text Preview (Total Words: {total_words_count}):**")
                        preview_text_html = full_raw_text.replace('\n', '<br>')
                        st.markdown(f"""
                        <div style="height: 180px; overflow-y: scroll; border: 1px solid rgba(255,255,255,0.1); padding: 10px; border-radius: 6px; font-size: 0.85rem; background-color: rgba(255,255,255,0.02); line-height: 1.5; color: #E2E8F0; margin-bottom: 15px;">
                            {preview_text_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        start_w, end_w = st.slider(
                            "Select Word Range:",
                            min_value=1,
                            max_value=total_words_count,
                            value=(1, min(500, total_words_count)),
                            step=1,
                            key="diffit_manual_range_slider"
                        )
                        
                        col_start, col_end = st.columns(2)
                        with col_start:
                            start_input = st.number_input("Start Word:", min_value=1, max_value=total_words_count, value=start_w, key="diffit_manual_start_input")
                        with col_end:
                            end_input = st.number_input("End Word:", min_value=1, max_value=total_words_count, value=end_w, key="diffit_manual_end_input")
                            
                        # Correct range values if inverted
                        chosen_start = min(start_input, end_input)
                        chosen_end = max(start_input, end_input)
                
                # If Automatically Summarise, select method
                summarise_method = "Traditional Extractive"
                algorithm = "Luhn"
                if extraction_method == "Automatically Summarise":
                    summarise_method = st.radio(
                        "Summarisation Method",
                        ["Traditional Extractive", "AI-Powered Abstractive"],
                        key="diffit_summarise_method"
                    )
                    if summarise_method == "Traditional Extractive":
                        st.write("**Extractive Algorithm:**")
                        col_luhn, col_lex = st.columns(2)
                        col_lsa, col_kl = st.columns(2)
                        with col_luhn:
                            algo_luhn = st.checkbox("Luhn", value=True, key="diffit_algo_luhn", on_change=select_luhn)
                        with col_lex:
                            algo_lexrank = st.checkbox("LexRank", value=False, key="diffit_algo_lexrank", on_change=select_lexrank)
                        with col_lsa:
                            algo_lsa = st.checkbox("Lsa", value=False, key="diffit_algo_lsa", on_change=select_lsa)
                        with col_kl:
                            algo_kl = st.checkbox("KL", value=False, key="diffit_algo_kl", on_change=select_kl)
                        
                        selected_algos = []
                        if algo_luhn: selected_algos.append("Luhn")
                        if algo_lexrank: selected_algos.append("LexRank")
                        if algo_lsa: selected_algos.append("Lsa")
                        if algo_kl: selected_algos.append("KL")
                        
                        algorithm = selected_algos[0] if selected_algos else "Luhn"
                
                st.markdown("---")
                
                # Extract button
                is_extract_disabled = (extraction_method == "Select Text Parts" and random_checked)
                if st.button("⚙️ Extract & Prepare Text", type="primary", use_container_width=True, disabled=is_extract_disabled):
                    # Resolve document choice
                    loaded_docs = []
                    if random_checked:
                        if files:
                            import random
                            chosen = random.choice(files)
                            loaded_docs = [chosen]
                        else:
                            loaded_docs = []
                    else:
                        loaded_docs = selected_files
                            
                    if not loaded_docs:
                        st.error("⚠️ Please select at least one document or check '[Random Document]'.")
                    else:
                        with st.spinner("Extracting text from document(s)..."):
                            # Get the document text(s)
                            texts = []
                            for doc in loaded_docs:
                                txt = get_text_data(db_path, "By File Name", "filename", doc)
                                if txt:
                                    texts.append(txt)
                            
                            if not texts:
                                st.error("Failed to load text for the selected document(s).")
                            else:
                                full_text = "\n\n".join(texts)
                                if extraction_method == "Automatically Summarise":
                                    from core.modules.diffit_generator import summarise_source_text
                                    extracted = summarise_source_text(
                                        text=full_text,
                                        method=summarise_method,
                                        algorithm=algorithm,
                                        word_limit=target_words,
                                        ai_provider=get_state('ai_provider'),
                                        ollama_model=get_state('ai_model'),
                                        api_key=get_state('gemini_api_key'),
                                        ollama_url=get_state('ollama_url'),
                                        gemini_model=get_state('gemini_model')
                                    )
                                elif extraction_method == "Chunk Text As Is":
                                    from core.modules.diffit_generator import extract_random_slice
                                    extracted = extract_random_slice(full_text, target_words)
                                else:
                                    # Select Text Parts
                                    word_list = full_text.split()
                                    sliced_words = word_list[chosen_start-1:chosen_end]
                                    extracted = " ".join(sliced_words)
                                    
                                if extracted:
                                    set_state('setting_mode_extracted_text', extracted)
                                    set_state('setting_mode_extracted_doc', ", ".join(loaded_docs))
                                    st.success("Text extracted successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to extract or summarise the text.")

            with col_preview:
                extracted_passage = get_state('setting_mode_extracted_text')
                doc_name = get_state('setting_mode_extracted_doc')
                
                if not extracted_passage:
                    st.subheader("📝 Text Preview Workspace")
                    st.markdown("""
                        <div style="border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 40px; text-align: center; color: #64748B;">
                            <span style="font-size: 50px;">📄</span>
                            <p style="margin-top: 15px; font-size: 1rem; color: #94A3B8;">No text extracted yet.</p>
                            <p style="font-size: 0.85rem;">Configure the text extraction parameters on the left and click <b>⚙️ Extract & Prepare Text</b> to begin.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.subheader("📝 Text Preview Workspace")
                    st.caption(f"Source Document: `{doc_name}`")
                    
                    # Editable text area
                    edited = st.text_area(
                        "Extracted Reading Passage (Editable)",
                        value=extracted_passage,
                        height=300,
                        key="diffit_edit_extracted_text"
                    )
                    if edited != extracted_passage:
                        set_state('setting_mode_extracted_text', edited)
                        
                    st.markdown("---")
                    st.markdown("### 🛠️ Step 2: Choose Quiz Generation Types")
                    
                    # Quiz type configuration checkboxes
                    st.checkbox("📚 Reading Comprehension Questions (MCQ & Short Answer)", key="diffit_quiz_reading")
                    st.checkbox("📝 Vocabulary Exercises (Spelling & POS Distractors)", key="diffit_quiz_grammar")
                    st.checkbox("🔤 Vocabulary Quizzes (Definitions & Example matches)", key="diffit_quiz_vocab")
                    
                    has_selected_quiz = get_state("diffit_quiz_reading") or get_state("diffit_quiz_grammar") or get_state("diffit_quiz_vocab")
                    
                    if has_selected_quiz:
                        if st.button("🚀 Generate Selected Activities", type="primary", use_container_width=True):
                            if get_state("diffit_quiz_grammar"):
                                with st.spinner("Generating vocabulary exercises..."):
                                    from core.modules.diffit_generator import generate_grammar_exercises
                                    grammar_res = generate_grammar_exercises(
                                        passage=extracted_passage,
                                        ai_provider=get_state('ai_provider'),
                                        ollama_model=get_state('ai_model'),
                                        api_key=get_state('gemini_api_key'),
                                        ollama_url=get_state('ollama_url'),
                                        gemini_model=get_state('gemini_model')
                                    )
                                    if grammar_res:
                                        set_state('setting_mode_grammar_exercises', grammar_res)
                                        st.success("Vocabulary exercises generated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to generate vocabulary exercises.")
                                        
                            if get_state("diffit_quiz_reading"):
                                with st.spinner("Generating reading comprehension questions..."):
                                    from core.modules.diffit_generator import generate_reading_comprehension
                                    reading_res = generate_reading_comprehension(
                                        passage=extracted_passage,
                                        ai_provider=get_state('ai_provider'),
                                        ollama_model=get_state('ai_model'),
                                        api_key=get_state('gemini_api_key'),
                                        ollama_url=get_state('ollama_url'),
                                        gemini_model=get_state('gemini_model')
                                    )
                                    if reading_res:
                                        set_state('setting_mode_reading_exercises', reading_res)
                                        st.success("Reading comprehension questions generated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to generate reading comprehension questions.")
                                        
                        if get_state("diffit_quiz_vocab"):
                            st.info("ℹ️ Note: Vocabulary quiz generation will be connected in subsequent steps.")
                    
                    # If we have generated grammar exercises in state, display them interactively!
                    grammar_exercises = get_state('setting_mode_grammar_exercises')
                    if isinstance(grammar_exercises, dict) and get_state("diffit_quiz_grammar"):
                        st.markdown("---")
                        st.subheader("📝 Generated Vocabulary Exercises")
                        
                        # Part 1: Contextual Fill-in-the-Blanks (Cloze)
                        if "fill_in_blanks" in grammar_exercises and isinstance(grammar_exercises["fill_in_blanks"], list):
                            with st.expander("📝 Part 1: Spelling & Word Completion Exercises (5 Items)", expanded=True):
                                st.markdown("*Complete the sentences by typing the correct spelling of the gapped words.*")
                                for i, item in enumerate(grammar_exercises["fill_in_blanks"]):
                                    if not isinstance(item, dict):
                                        st.markdown(f"**Item {i+1}**: {item}")
                                        st.markdown("---")
                                        continue
                                    
                                    sentence = item.get('sentence', 'Missing sentence')
                                    st.markdown(f"**Item {i+1}:** {sentence}")
                                    user_ans = st.text_input(
                                        f"Your Answer for Item {i+1} (gap: {item.get('base_word', '')}):",
                                        key=f"diffit_grammar_cloze_{i}"
                                    )
                                    if user_ans:
                                        correct = item.get('correct_answer', '').strip().lower()
                                        if user_ans.strip().lower() == correct:
                                            st.success("🎉 Correct!")
                                        else:
                                            st.error(f"❌ Incorrect. Expected: **{item.get('correct_answer', '')}**")
                                             
                                    with st.popover(f"🔍 Show Answer & Explanation (Item {i+1})"):
                                        st.markdown(f"**Correct Answer:** `{item.get('correct_answer', '')}`")
                                        st.markdown(f"**Explanation:** {item.get('explanation', 'None')}")
                                    st.markdown("---")
                                     
                        # Part 2: True/False Vocabulary & Grammar Exercises
                        if "true_false" in grammar_exercises and isinstance(grammar_exercises["true_false"], list):
                            with st.expander("❓ Part 2: True / False Vocabulary Exercises (5 Items)", expanded=True):
                                st.markdown("*Decide whether the highlighted word in each sentence is correct based on the reading passage.*")
                                for i, item in enumerate(grammar_exercises["true_false"]):
                                    if not isinstance(item, dict):
                                        st.markdown(f"**Item {i+1}**: {item}")
                                        st.markdown("---")
                                        continue
                                    
                                    sentence = item.get('sentence', 'Missing sentence')
                                    st.markdown(f"**Item {i+1}:** {sentence}", unsafe_allow_html=True)
                                    
                                    user_ans = st.radio(
                                        f"Is the highlighted word correct for Item {i+1}?",
                                        ["Select...", "True", "False"],
                                        key=f"diffit_grammar_tf_{i}"
                                    )
                                    
                                    if user_ans != "Select...":
                                        user_bool = (user_ans == "True")
                                        correct_bool = item.get('is_correct', True)
                                        if user_bool == correct_bool:
                                            st.success("🎉 Correct!")
                                        else:
                                            st.error("❌ Incorrect.")
                                            
                                    with st.popover(f"🔍 Show Answer & Explanation (Item {i+1})"):
                                        st.markdown(f"**Correct Answer:** `{item.get('is_correct', True)}`")
                                        st.markdown(f"**Explanation:** {item.get('explanation', 'None')}")
                                    st.markdown("---")
                                     
                        # Part 3: Multiple-Choice Vocabulary Questions
                        if "mcqs" in grammar_exercises and isinstance(grammar_exercises["mcqs"], list):
                            with st.expander("❓ Part 3: Multiple-Choice Vocabulary Questions (5 Items)", expanded=True):
                                st.markdown("*Select the correct grammatical option to complete the passage-based question.*")
                                for i, item in enumerate(grammar_exercises["mcqs"]):
                                    if not isinstance(item, dict):
                                        st.markdown(f"**Item {i+1}**: {item}")
                                        st.markdown("---")
                                        continue
                                    
                                    question = item.get('question', 'Missing question')
                                    if "[Blank]" in question:
                                        question = question.replace("[Blank]", '<span class="premium-blank-box"></span>')
                                    st.markdown(f"**Item {i+1}:** {question}", unsafe_allow_html=True)
                                    
                                    options_dict = item.get('options', {})
                                    options_list = sorted(options_dict.keys()) # ['A', 'B', 'C', 'D', 'E']
                                    
                                    # Display option choices
                                    for letter in options_list:
                                        st.markdown(f"**{letter}:** {options_dict[letter]}")
                                    
                                    user_ans = st.radio(
                                        f"Select Option for Item {i+1}:",
                                        ["Select..."] + options_list,
                                        key=f"diffit_grammar_mcq_{i}",
                                        label_visibility="collapsed"
                                    )
                                    
                                    if user_ans != "Select...":
                                        correct_letter = item.get('correct_option', 'A')
                                        if user_ans == correct_letter:
                                            st.success(f"🎉 Correct! The answer is {correct_letter} ({options_dict.get(correct_letter, '')}).")
                                        else:
                                            st.error("❌ Incorrect. Try again.")
                                            
                                    with st.popover(f"🔍 Show Answer & Explanation (Item {i+1})"):
                                        correct_letter = item.get('correct_option', 'A')
                                        st.markdown(f"**Correct Option:** `{correct_letter}`")
                                        st.markdown(f"**Correct Word:** `{options_dict.get(correct_letter, '')}`")
                                        st.markdown(f"**Explanation:** {item.get('explanation', 'None')}")
                                    st.markdown("---")
                     
                    # If we have generated reading comprehension exercises in state, display them interactively!
                    reading_exercises = get_state('setting_mode_reading_exercises')
                    if reading_exercises and get_state("diffit_quiz_reading"):
                        questions = []
                        type_4_skipped = False
                        type_5_skipped = False
                        if isinstance(reading_exercises, dict):
                            questions = reading_exercises.get("questions", [])
                            type_4_skipped = reading_exercises.get("type_4_skipped", False)
                            type_5_skipped = reading_exercises.get("type_5_skipped", False)
                        elif isinstance(reading_exercises, list):
                            questions = reading_exercises
                            
                        st.markdown("---")
                        st.subheader("📚 Generated Reading Comprehension Questions")
                        
                        if type_4_skipped:
                            st.warning("⚠️ **Paragraph Completion Questions (Type 4) Skipped:** The text has less than 6 sentences, so paragraph start/end questions could not be generated.")
                            
                        if type_5_skipped:
                            st.warning("⚠️ **Named Entity Questions (Type 5) Skipped:** The text does not contain any recognized named entities (names, locations, dates, etc.). If you want to include Type 5 questions, please provide a text or upload a corpus containing these entities.")
                            
                        with st.expander("📚 Part 1: Reading Comprehension Exercises (10 Items)", expanded=True):
                            st.markdown("*Select the correct option based on the reading passage.*")
                            for i, item in enumerate(questions):
                                if not isinstance(item, dict):
                                    continue
                                
                                question = item.get('question', '')
                                q_type = item.get('type', '')
                                
                                if q_type in ["type_4_start", "type_4_end"] and "\n\n> " in question:
                                    parts = question.split("\n\n> ")
                                    prompt_instruction, segment_text = parts[0], parts[1]
                                    st.markdown(f"**Question {i+1}:** {prompt_instruction}")
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: rgba(0, 173, 181, 0.05); 
                                            border-left: 4px solid #00ADB5; 
                                            padding: 12px 16px; 
                                            border-radius: 6px; 
                                            margin: 10px 0 15px 0;
                                            font-style: italic;
                                            line-height: 1.6;
                                            color: #E0E0E0;
                                        ">
                                            {segment_text}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                else:
                                    if "[Blank]" in question:
                                        question_html = question.replace("[Blank]", '<span class="premium-blank-box"></span>')
                                        st.markdown(f"**Question {i+1}:** {question_html}", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"**Question {i+1}:** {question}")
                                
                                options_dict = item.get('options', {})
                                options_list = sorted(options_dict.keys())
                                
                                for letter in options_list:
                                    st.markdown(f"**{letter}:** {options_dict[letter]}")
                                    
                                user_ans = st.radio(
                                    f"Select Option for Question {i+1}:",
                                    ["Select..."] + options_list,
                                    key=f"diffit_reading_ans_{i}",
                                    label_visibility="collapsed"
                                )
                                
                                if user_ans != "Select...":
                                    correct_letter = item.get('correct_option', 'A')
                                    if user_ans == correct_letter:
                                        st.success(f"🎉 Correct! The answer is {correct_letter}.")
                                    else:
                                        st.error(f"❌ Incorrect. Try again.")
                                        
                                with st.popover(f"🔍 Show Answer & Explanation (Question {i+1})"):
                                    correct_letter = item.get('correct_option', 'A')
                                    st.markdown(f"**Correct Option:** `{correct_letter}`")
                                    st.markdown(f"**Correct Answer:** `{options_dict.get(correct_letter, '')}`")
                                    st.markdown(f"**Explanation / Distractor Details:**\n{item.get('explanation', 'None')}")
                                st.markdown("---")
                    
                    st.markdown("---")
                    if st.button("❌ Reset"):
                        set_state('setting_mode_extracted_text', None)
                        set_state('setting_mode_extracted_doc', None)
                        set_state('setting_mode_grammar_exercises', None)
                        set_state('setting_mode_reading_exercises', None)
                        st.rerun()

        # =====================================================================
        # TAB: AUTOMATIC MODE (FULLY IMPLEMENTED)
        # =====================================================================
        with tab_auto:
            # State key for currently generated quiz
            quiz_key = f"generated_quiz_{db_path}"
            generated_quiz = get_state(quiz_key)

            # Automatic stale cache invalidation:
            # If the stored quiz in st.session_state is in the old format (e.g. single-word answers in Section D),
            # clear it immediately to force the user to see the clean state and generate a fresh one.
            try:
                if generated_quiz:
                    # Invalidate if Section E is missing or Section D is in the old single-word format
                    is_old = 'section_e' not in generated_quiz or any(' ' not in q.get('expected_answer', '') for q in generated_quiz.get('section_d', []))
                    if is_old:
                        set_state(quiz_key, None)
                        generated_quiz = None
            except Exception as e:
                # Safely log and ignore cache errors to prevent UI block
                print(f"Quiz cache invalidator error: {e}")

            st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #FFFFFF; font-size: 0.95rem;">Active Corpus: <b style="color: #00FFF5;">{corpus_name}</b></span>
                    <span style="background-color: #00ADB5; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase;">100% Offline</span>
                </div>
            """, unsafe_allow_html=True)

            # Render XML restriction filters
            xml_filters = render_xml_restriction_filters(db_path, "quiz", corpus_name=corpus_name)
            xml_where, xml_params = apply_xml_restrictions(xml_filters)

            col_ctrls, col_preview = st.columns([1, 3])

            with col_ctrls:
                st.markdown("### 🛠️ Controls")

                # Action: Generate Quiz
                if st.button("🚀 Generate Quiz", type="primary", use_container_width=True):
                    with st.spinner("Analyzing corpus & generating 20 items..."):
                        quiz = generate_full_quiz(db_path, xml_where_clause=xml_where, xml_params=xml_params)
                        if quiz.get('success'):
                            quiz['xml_filters'] = xml_filters
                            set_state(quiz_key, quiz)
                            st.success("Quiz generated successfully!")
                            st.rerun()
                        else:
                            st.error(quiz.get('error', "Failed to generate quiz."))

                if generated_quiz:
                    st.markdown("---")
                    st.markdown("#### 🔄 Regenerate Section")
                    section_to_regen = st.radio(
                        "Select Section",
                        ["Section A (Discourse)", "Section B (Reordering)", "Section C (Grammar)", "Section D (MWE)"],
                        horizontal=True,
                        key="select_section_to_regen"
                    )

                    if st.button("Regenerate Section", use_container_width=True):
                        with st.spinner(f"Regenerating {section_to_regen}..."):
                            sentences = get_corpus_sentences(db_path, xml_where_clause=xml_where, xml_params=xml_params)

                            if section_to_regen.startswith("Section A"):
                                section_a = generate_section_a(sentences, num_passages=2)
                                if section_a:
                                    generated_quiz['section_a'] = section_a
                                    set_state(quiz_key, generated_quiz)
                                    st.success("Section A regenerated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate suitable passages. Try a different corpus.")

                            elif section_to_regen.startswith("Section B"):
                                section_b = generate_section_b(sentences, generated_quiz.get('section_a', []), num_items=5)
                                if section_b:
                                    generated_quiz['section_b'] = section_b
                                    set_state(quiz_key, generated_quiz)
                                    st.success("Section B regenerated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate sentence ordering items.")

                            elif section_to_regen.startswith("Section C"):
                                used_texts = set()
                                for p in generated_quiz.get('section_a', []):
                                    used_texts.add(p['original_passage'])
                                for item in generated_quiz.get('section_b', []):
                                    for s in item['original_sentences']:
                                        used_texts.add(s)

                                section_c = generate_section_c(sentences, used_texts, num_questions=5)
                                if section_c:
                                    generated_quiz['section_c'] = section_c
                                    set_state(quiz_key, generated_quiz)
                                    st.success("Section C regenerated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate grammar questions.")

                            elif section_to_regen.startswith("Section D"):
                                used_texts = set()
                                for p in generated_quiz.get('section_a', []):
                                    used_texts.add(p['original_passage'])
                                for item in generated_quiz.get('section_b', []):
                                    for s in item['original_sentences']:
                                        used_texts.add(s)
                                for q in generated_quiz.get('section_c', []):
                                    used_texts.add(q['original_sentence'])

                                section_d = generate_section_d(
                                    db_path, sentences, used_texts, num_questions=5,
                                    xml_where_clause=xml_where, xml_params=xml_params
                                )
                                if section_d:
                                    generated_quiz['section_d'] = section_d
                                    set_state(quiz_key, generated_quiz)
                                    st.success("Section D regenerated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate collocation questions.")

                    st.markdown("---")
                    st.markdown("#### 📥 Export & Download")

                    # Timestamp for unique filenames
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Generate DOCX buffers
                    try:
                        exercises_io = create_exercises_docx(generated_quiz, corpus_name)
                        st.download_button(
                            label="📄 Exercises (.docx)",
                            data=exercises_io.getvalue(),
                            file_name=f"cortex_exercises_{ts}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating exercise DOCX: {e}")

                    try:
                        answer_key_io = create_answer_key_docx(generated_quiz, corpus_name)
                        st.download_button(
                            label="🔑 Answer Key (.docx)",
                            data=answer_key_io.getvalue(),
                            file_name=f"cortex_answer_key_{ts}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating answer key DOCX: {e}")

            # PREVIEW AREA (RIGHT COLUMN)
            with col_preview:
                if not generated_quiz:
                    st.subheader("📝 Preview Mode")
                    st.markdown("""
                        <div style="border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 40px; text-align: center; color: #64748B;">
                            <span style="font-size: 50px;">📄</span>
                            <p style="margin-top: 15px; font-size: 1rem; color: #94A3B8;">No quiz generated yet.</p>
                            <p style="font-size: 0.85rem;">Click the <b>🚀 Generate Quiz</b> button on the left to instantly extract 20 interactive items from your corpus.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("### 📝 Interactive Quiz Preview")

                    # Active filters descriptor
                    active_filters_desc = []
                    q_filters = generated_quiz.get('xml_filters')
                    if q_filters:
                        for k, v in q_filters.items():
                            if v['type'] == 'list':
                                active_filters_desc.append(f"**{k.capitalize()}**: {', '.join(v['values'])}")
                            elif v['type'] == 'range':
                                active_filters_desc.append(f"**{k.capitalize()}**: {v['min']} - {v['max']}")

                    if active_filters_desc:
                        st.info(f"💡 **Sub-corpus filters applied:** {'; '.join(active_filters_desc)}")

                    # Render Section A
                    with st.expander("📌 SECTION A — Discourse Completion (10 items)", expanded=True):
                        st.markdown("""
                            <i><b>Instructions:</b> Read the following passages. Five sentences have been removed from each passage and replaced with markers. 
                            Match the shuffled options below the passage to their correct missing sentence location.</i>
                        """, unsafe_allow_html=True)

                        for idx, passage in enumerate(generated_quiz['section_a'], start=1):
                            st.markdown(f"#### Passage {idx}")
                            # Display gapped passage with highlights
                            highlighted_gapped = passage['gapped_passage']
                            for letter in ['A', 'B', 'C', 'D', 'E']:
                                marker = f"[MISSING SENTENCE {letter}]"
                                highlighted_gapped = highlighted_gapped.replace(
                                    marker, 
                                    f"<b style='color:#00FFF5; background-color:rgba(0,173,181,0.2); padding: 2px 6px; border-radius: 4px;'>{marker}</b>"
                                )

                            st.markdown(f"""
                                <div style="background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); line-height: 1.6; text-align: justify; margin-bottom: 15px;">
                                    {highlighted_gapped}
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("<b>Removed sentences (Shuffled):</b>", unsafe_allow_html=True)
                            for opt in passage['options']:
                                st.markdown(f"**{opt['letter']}.** {opt['sentence_text']}")

                            # Self-check helper
                            with st.popover(f"🔍 Show Passage {idx} Answers"):
                                st.markdown("##### Correct Mappings:")
                                for marker in ['[MISSING SENTENCE A]', '[MISSING SENTENCE B]', '[MISSING SENTENCE C]', '[MISSING SENTENCE D]', '[MISSING SENTENCE E]']:
                                    correct_letter = passage['correct_mapping'].get(marker)
                                    st.markdown(f"**{marker}** = Option **{correct_letter}**")
                            st.markdown("---")

                    # Render Section B
                    with st.expander("📌 SECTION B — Sentence Reordering (5 items)", expanded=False):
                        st.markdown("""
                            <i><b>Instructions:</b> The sentences in each item have been extracted from the corpus and randomized. 
                            Reorder them into their original coherent sequence.</i>
                        """, unsafe_allow_html=True)

                        for idx, item in enumerate(generated_quiz['section_b'], start=1):
                            st.markdown(f"#### Item {idx}")

                            for s_idx, sent in enumerate(item['randomized_sentences'], start=1):
                                st.markdown(f"**[{s_idx}]** {sent}")

                            # User typing check
                            user_order = st.text_input(
                                f"Your Order Sequence (e.g. 3-1-5-2-4):",
                                key=f"b_user_input_{idx}",
                                placeholder="Type sequence here..."
                            )

                            if user_order:
                                if user_order.strip() == item['correct_sequence']:
                                    st.success("🎉 Correct sequence!")
                                else:
                                    st.error("❌ Incorrect sequence. Try again.")

                            with st.popover(f"🔍 Show Item {idx} Answer"):
                                st.markdown(f"Correct sequence order: **{item['correct_sequence']}**")
                                st.markdown("**Original Order Sequence:**")
                                for s_idx, sent in enumerate(item['original_sentences'], start=1):
                                    st.markdown(f"**[{s_idx}]** {sent}")
                            st.markdown("---")

                    # Render Section C
                    with st.expander("📌 SECTION C — Grammar Questions (5 items)", expanded=False):
                        st.markdown("""
                            <i><b>Instructions:</b> Select the grammatically correct option (A-E) to complete each sentence.</i>
                        """, unsafe_allow_html=True)

                        for idx, q in enumerate(generated_quiz['section_c'], start=1):
                            st.markdown(f"**{idx}.** {q['prompt']} *(Type: {q['type']})*")

                            cols = st.columns(5)
                            for o_idx, opt in enumerate(q['options']):
                                cols[o_idx].markdown(f"**{opt['letter']})** {opt['text']}")

                            user_ans = st.radio(
                                f"Select Option for Question {idx}:",
                                ['Select...', 'A', 'B', 'C', 'D', 'E'],
                                key=f"c_radio_{idx}",
                                label_visibility="collapsed"
                            )

                            if user_ans != 'Select...':
                                if user_ans == q['correct_letter']:
                                    st.success(f"🎉 Correct! The answer is {q['correct_letter']} ({q['correct_answer']}).")
                                else:
                                    st.error(f"❌ Incorrect. Try again.")

                            with st.popover(f"🔍 Show Question {idx} Answer"):
                                st.markdown(f"Correct option: **{q['correct_letter']} ({q['correct_answer']})**")
                                st.markdown(f"**Full Context Sentence:** *{q['original_sentence']}*")
                            st.markdown("---")

                    # Render Section D
                    with st.expander("📌 SECTION D — Multiword Expression Questions (5 items)", expanded=False):
                        st.markdown("""
                            <i><b>Instructions:</b> Complete the following sentences or expressions with the appropriate corpus-driven collocation (both words).</i>
                        """, unsafe_allow_html=True)

                        for idx, q in enumerate(generated_quiz['section_d'], start=1):
                            st.markdown(f"**{idx}.** {q['prompt']} *(Collocation Pattern: {q['type']})*")

                            user_ans = st.text_input(
                                f"Enter answer for MWE {idx} (both words):",
                                key=f"d_input_{idx}",
                                placeholder="Type both words here..."
                            )

                            if user_ans:
                                if user_ans.strip().lower() == q['expected_answer'].lower():
                                    st.success("🎉 Correct!")
                                else:
                                    st.error(f"❌ Incorrect. Expected: **{q['expected_answer']}**")

                            with st.popover(f"🔍 Show MWE {idx} Answer"):
                                st.markdown(f"Expected Collocation: **{q['expected_answer']}**")
                                st.markdown(f"Full Collocation: ***{q['collocation']}***")
                                st.markdown(f"**Context Sentence:** *{q['original_sentence']}*")
                            st.markdown("---")

                    # Render Section E
                    with st.expander("📌 SECTION E — Sentence Composition (5 items)", expanded=False):
                        st.markdown("""
                            <i><b>Instructions:</b> Reorganize the following scrambled words into a grammatically correct and well-formed sentence.</i>
                        """, unsafe_allow_html=True)

                        import re
                        def clean_sentence_compare(s1, s2):
                            w1 = re.findall(r'\b\w+\b', str(s1).lower())
                            w2 = re.findall(r'\b\w+\b', str(s2).lower())
                            return w1 == w2

                        for idx, q in enumerate(generated_quiz.get('section_e', []), start=1):
                            st.markdown(f"**{idx}.** {q['prompt']}")

                            user_ans = st.text_input(
                                f"Enter reconstructed sentence for Question {idx}:",
                                key=f"e_input_{idx}",
                                placeholder="Type sentence here..."
                            )

                            if user_ans:
                                if clean_sentence_compare(user_ans, q['expected_answer']):
                                    st.success("🎉 Correct sentence composition!")
                                else:
                                    st.error("❌ Incorrect sentence structure. Try again.")

                            with st.popover(f"🔍 Show Scrambled Sentence {idx} Answer"):
                                st.markdown(f"Correct Reorganized Sentence: **{q['expected_answer']}**")
                            st.markdown("---")

