import streamlit as st
import datetime
import importlib
from ui_streamlit.state_manager import get_state, set_state
from ui_streamlit.components.filters import render_xml_restriction_filters
from core.preprocessing.xml_parser import apply_xml_restrictions

import core.modules.quiz_creation
importlib.reload(core.modules.quiz_creation)
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
    st.markdown("""
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
            st.subheader("⚙️ Quiz Generation Settings")
            st.info("ℹ️ Custom Quiz configuration options will be available here soon.")

            st.markdown("""
                <div style="background-color: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 12px; border: 1px dashed rgba(255, 255, 255, 0.1); margin-top: 15px;">
                    <h4 style="color: #00FFF5; margin-top: 0;">🚧 Under Construction</h4>
                    <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">
                        The Settings Mode is currently under construction. In future updates, you will be able to customize:
                    </p>
                    <ul style="color: #F1F5F9; line-height: 1.8; font-size: 0.9rem; margin-left: 20px;">
                        <li><b>CEFR Difficulty Level:</b> Target A1-C2 level vocabularies and structures.</li>
                        <li><b>Topic & Genre Filtering:</b> Restrict source texts to academic, news, fiction, etc.</li>
                        <li><b>Question Type Selector:</b> Choose which grammar patterns or parts of speech to emphasize.</li>
                        <li><b>AI-Assisted Generation:</b> Leverage local or cloud AI models to generate distractors and inferential questions.</li>
                        <li><b>Pedagogical taxonomies:</b> Filter questions based on Bloom's Taxonomy (Knowledge, Analysis, etc.).</li>
                    </ul>
                    <p style="color: #00ADB5; font-weight: bold; margin-bottom: 0; margin-top: 15px;">Stay tuned for more premium features!</p>
                </div>
            """, unsafe_allow_html=True)

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

