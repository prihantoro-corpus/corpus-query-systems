import unittest
import os
import sys
import tempfile
import duckdb
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.modules.quiz_creation import (
    generate_full_quiz,
    get_corpus_sentences,
    create_exercises_docx,
    create_answer_key_docx
)

class TestQuizCreation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary database for testing
        cls.db_fd, cls.db_path = tempfile.mkstemp(suffix='.duckdb')
        os.close(cls.db_fd)
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        
        # Populate with 45 mock sentences containing distinct grammatical structures, transitions, and collocations
        data = []
        
        # We need typical words to trigger Section C (Articles, Prepositions, Modals, SV agreement, Relatives, Passive)
        # We also need transitions like "however", "therefore" for Section A and B scoring.
        # We also need collocations like "significant difference" or "look after" for Section D.
        
        sentences_templates = [
            # Passage 1 (20 sentences)
            "The researchers conducted a new study on language acquisition.",
            "This result is extremely interesting because it shows high lexical cohesion.",
            "However, many participants could not complete the second task on time.",
            "Therefore, the final dataset consisted of only twenty speakers.",
            "The students who participated in this experiment were very motivated.",
            "The data were collected over a six-month period by a professional team.",
            "An interesting point is that they showed a significant difference in performance.",
            "This suggests that modal verbs can be acquired very early by child learners.",
            "In addition, child language looks very complex during the initial stage.",
            "Most children can learn any language that is spoken around them easily.",
            "This findings will be discussed in detail during the next conference.",
            "Meanwhile, the researchers are preparing another set of experiments.",
            "They will focus on relative clauses and passive voice constructions.",
            "These findings might change how teachers design pedagogical materials offline.",
            "Finally, the study concludes that natural corpus data is extremely valuable.",
            "Everyone agrees that authentic texts provide a rich source of context.",
            "Several studies have reported similar patterns in adult learners.",
            "But the current research goes a step further in analyzing these structures.",
            "It is highly important to understand the cognitive mechanism behind this.",
            "The next section will outline the research methodology and design.",
            
            # Passage 2 (20 sentences)
            "Language learning is a lifelong process that requires constant practice.",
            "Many learners struggle with prepositions and subject-verb agreement initially.",
            "Subsequently, they become more fluent as they read more corpus texts.",
            "This result was confirmed by several independent testing centers globally.",
            "The teachers who design the curricula must select appropriate materials.",
            "Authentic materials are preferred because they avoid artificial structures.",
            "Therefore, teachers often rely on corpora to extract common collocations.",
            "For example, they look after the frequency of specific multiword expressions.",
            "This study provides a practical guide on how to integrate corpora.",
            "It outlines five simple steps that can be followed easily.",
            "First, the teacher should identify the target grammar patterns.",
            "Second, they must query the corpus database to find examples.",
            "Next, they can generate multiple-choice questions automatically.",
            "Finally, the exercises must be exported to a professional Word document.",
            "This method has been proven to increase student motivation significantly.",
            "Students love working with real sentences instead of artificial examples.",
            "The results were published in a reputable journal last month.",
            "They showed that students made highly significant progress in grammar.",
            "We hope that more institutions will adopt this corpus-driven approach.",
            "In conclusion, offline corpus query systems are extremely beneficial."
        ]
        
        # Tokenize simply for database mock
        sent_id = 1
        global_token_id = 1
        for s_text in sentences_templates:
            tokens = [t.strip(",.?!;:") for t in s_text.split() if t.strip()]
            for t in tokens:
                t_low = t.lower()
                # Determine simple pos
                pos = "NN"
                if t_low in {"is", "are", "was", "were", "been", "be", "being", "has", "have", "had", "do", "does", "did"}:
                    pos = "AUX"
                elif t_low in {"conducted", "shows", "consisted", "participated", "collected", "suggests", "looks", "learn", "discussed", "preparing", "focus", "change", "design", "concludes", "provide", "reported", "goes", "understand", "outline", "requires", "struggle", "become", "confirmed", "rely", "integrate", "follow", "identify", "query", "find", "generate", "exported", "proven", "increase", "love", "working", "published", "made", "adopt", "hope"}:
                    pos = "VB"
                elif t_low in {"a", "an", "the"}:
                    pos = "DT"
                elif t_low in {"on", "in", "of", "by", "during", "around", "after", "with", "instead", "last"}:
                    pos = "IN"
                elif t_low in {"who", "that", "which"}:
                    pos = "WP"
                elif t_low in {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}:
                    pos = "MD"
                elif t_low in {"however", "therefore", "subsequently", "meanwhile", "finally", "first", "second", "next", "conconclusion"}:
                    pos = "RB"
                elif t_low in {"interesting", "lexical", "final", "motivated", "significant", "child", "complex", "initial", "pedagogical", "valuable", "authentic", "rich", "similar", "reputable", "beneficial"}:
                    pos = "JJ"
                
                # Check for passive VBN form
                if t_low in {"collected", "discussed", "published", "confirmed", "proven", "exported"}:
                    pos = "VBN"
                    
                data.append({
                    "id": global_token_id,
                    "token": t,
                    "pos": pos,
                    "lemma": t_low,
                    "sent_id": sent_id,
                    "filename": "test_doc_1.txt" if sent_id <= 20 else "test_doc_2.txt",
                    "_token_low": t_low
                })
                global_token_id += 1
            sent_id += 1
            
        # Ingest to temp database
        con = duckdb.connect(cls.db_path)
        df_src = pd.DataFrame(data)
        con.execute("CREATE TABLE corpus AS SELECT * FROM df_src")
        con.close()

    @classmethod
    def tearDownClass(cls):
        # Cleanup temp database
        if os.path.exists(cls.db_path):
            try:
                os.remove(cls.db_path)
            except:
                pass

    def test_get_corpus_sentences(self):
        sents = get_corpus_sentences(self.db_path)
        self.assertEqual(len(sents), 40)
        self.assertEqual(sents[0]['sent_id'], 1)
        self.assertIn("researchers", sents[0]['tokens'])

    def test_generate_full_quiz(self):
        quiz = generate_full_quiz(self.db_path)
        self.assertTrue(quiz['success'])
        
        # Test Section A (Discourse Completion)
        self.assertEqual(len(quiz['section_a']), 2)
        for passage in quiz['section_a']:
            self.assertEqual(len(passage['removed_sentences']), 5)
            self.assertEqual(len(passage['options']), 5)
            self.assertEqual(len(passage['correct_mapping']), 5)
            
        # Test Section B (Sentence Reordering)
        self.assertEqual(len(quiz['section_b']), 5)
        for item in quiz['section_b']:
            self.assertEqual(len(item['randomized_sentences']), 5)
            self.assertEqual(len(item['original_sentences']), 5)
            self.assertTrue("-" in item['correct_sequence'])
            
        # Test Section C (Grammar Questions)
        self.assertEqual(len(quiz['section_c']), 5)
        for q in quiz['section_c']:
            self.assertEqual(len(q['options']), 5)
            self.assertTrue(q['correct_letter'] in {'A', 'B', 'C', 'D', 'E'})
            
        # Test Section D (Multiword Expression Questions)
        self.assertEqual(len(quiz['section_d']), 5)
        for q in quiz['section_d']:
            self.assertIn("expected_answer", q)
            self.assertIn("prompt", q)
            
        # Test Section E (Sentence Composition Scramble)
        self.assertEqual(len(quiz['section_e']), 5)
        for q in quiz['section_e']:
            self.assertIn("expected_answer", q)
            self.assertIn("prompt", q)
            self.assertIn("shuffled_words", q)

    def test_docx_export(self):
        quiz = generate_full_quiz(self.db_path)
        self.assertTrue(quiz['success'])
        
        # Exercises DOCX
        ex_io = create_exercises_docx(quiz, "Test Mock Corpus")
        self.assertIsNotNone(ex_io)
        self.assertTrue(len(ex_io.getvalue()) > 0)
        
        # Answer Key DOCX
        ak_io = create_answer_key_docx(quiz, "Test Mock Corpus")
        self.assertIsNotNone(ak_io)
        self.assertTrue(len(ak_io.getvalue()) > 0)

    def test_generate_full_quiz_restricted(self):
        con = duckdb.connect(self.db_path)
        try:
            con.execute("ALTER TABLE corpus ADD COLUMN reading_ease_level VARCHAR")
        except:
            pass
            
        try:
            # Update 30 sentences to have level = 'Easy'
            con.execute("UPDATE corpus SET reading_ease_level = 'Easy' WHERE sent_id <= 30")
            # Update the other 10 sentences to have level = 'Difficult'
            con.execute("UPDATE corpus SET reading_ease_level = 'Difficult' WHERE sent_id > 30")
        finally:
            con.close()
            
        # Test 1: Generate quiz restricted to 'Easy' level (should succeed since we have 30 sentences)
        quiz_easy = generate_full_quiz(self.db_path, xml_where_clause=" AND reading_ease_level = ?", xml_params=["Easy"])
        self.assertTrue(quiz_easy['success'])
        self.assertEqual(len(quiz_easy['section_a']), 2)
        
        # Test 2: Generate quiz restricted to 'Difficult' level (should fail because there are only 10 sentences)
        quiz_diff = generate_full_quiz(self.db_path, xml_where_clause=" AND reading_ease_level = ?", xml_params=["Difficult"])
        self.assertFalse(quiz_diff['success'])
        self.assertIn("at least 30 sentences are required", quiz_diff['error'])
        self.assertIn("only 10 sentences", quiz_diff['error'])

if __name__ == '__main__':
    unittest.main()
