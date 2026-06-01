import unittest
from core.modules.lexical_complexity import (
    classify_pos_tag,
    calculate_generic_complexity,
    calculate_specific_complexity
)

class TestLexicalComplexity(unittest.TestCase):
    
    def test_classify_pos_tag(self):
        # Case 1: English PTB fallbacks
        self.assertEqual(classify_pos_tag("NN", "", "English"), "N")
        self.assertEqual(classify_pos_tag("VBZ", "", "English"), "V")
        self.assertEqual(classify_pos_tag("JJ", "", "English"), "Adj")
        self.assertEqual(classify_pos_tag("RB", "", "English"), "Adv")
        
        # Case 2: Definition keywords (any language fallback)
        self.assertEqual(classify_pos_tag("XYZ", "singular noun", "OtherLanguage"), "N")
        self.assertEqual(classify_pos_tag("ABC", "transitive verb", "OtherLanguage"), "V")
        self.assertEqual(classify_pos_tag("PQR", "comparative adjective", "OtherLanguage"), "Adj")
        self.assertEqual(classify_pos_tag("MNO", "locative adverb", "OtherLanguage"), "Adv")
        
        # Case 3: Japanese BCCWJ Kanji keywords
        self.assertEqual(classify_pos_tag("名詞-普通名詞-一般", "", "Japanese"), "N")
        self.assertEqual(classify_pos_tag("動詞-一般", "", "Japanese"), "V")

    def test_calculate_generic_complexity(self):
        lemmas = ["the", "dog", "barks", "at", "the", "cat", ".", "123"]
        # Punctuation "." and numeric "123" are skipped
        # Remaining valid lemmas: ["the", "dog", "barks", "at", "the", "cat"]
        # N = 6, unique V = 5 ("the", "dog", "barks", "at", "cat")
        res = calculate_generic_complexity(lemmas)
        
        self.assertEqual(res["N"], 6)
        self.assertEqual(res["V"], 5)
        self.assertEqual(res["TTR"], round(5/6, 4))
        self.assertTrue(res["RTTR"] > 0)
        self.assertTrue(res["CTTR"] > 0)
        self.assertTrue(res["MTLD"] > 0)
        
    def test_calculate_specific_complexity(self):
        tokens = ["The", "dog", "barks", "quickly", "."]
        lemmas = ["the", "dog", "bark", "quickly", "."]
        pos_tags = ["DT", "NN", "VBZ", "RB", "SENT"]
        pos_definitions = {
            "DT": "determiner",
            "NN": "noun, singular",
            "VBZ": "verb, 3rd person singular present",
            "RB": "adverb",
            "SENT": "sentence-ending punctuation"
        }
        
        # Valid non-punctuation: "The", "dog", "barks", "quickly" (N=4)
        # Lexical words: "dog" (N), "bark" (V), "quickly" (Adv) (Lexical tokens = 3)
        res = calculate_specific_complexity(tokens, lemmas, pos_tags, pos_definitions, "English")
        
        self.assertEqual(res["LD"], round(3/4, 4)) # 3 content words / 4 total
        self.assertEqual(res["LV"], 1.0) # all lexical lemmas are unique
        self.assertEqual(res["NV"], round(1/3, 4)) # 1 noun type / 3 lexical tokens
        self.assertEqual(res["VV1"], 1.0) # 1 verb type / 1 verb token
        self.assertEqual(res["AdvV"], round(1/3, 4)) # 1 adverb type / 3 lexical tokens

    def test_calculate_corpus_lexical_complexity(self):
        import os
        import tempfile
        import duckdb
        from core.modules.lexical_complexity import calculate_corpus_lexical_complexity
        
        temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(temp_db_fd)
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        
        try:
            conn = duckdb.connect(temp_db_path)
            conn.execute("CREATE TABLE corpus (id INTEGER, filename VARCHAR, token VARCHAR, lemma VARCHAR, pos VARCHAR, genre VARCHAR)")
            conn.execute("INSERT INTO corpus VALUES (1, 'doc1.txt', 'The', 'the', 'DT', 'news')")
            conn.execute("INSERT INTO corpus VALUES (2, 'doc1.txt', 'cat', 'cat', 'NN', 'news')")
            conn.execute("INSERT INTO corpus VALUES (3, 'doc2.txt', 'slept', 'sleep', 'VBD', 'fiction')")
            conn.close()
            
            # Test group by default (filename)
            res_file = calculate_corpus_lexical_complexity(temp_db_path)
            self.assertIn("doc1.txt", res_file["files"])
            self.assertIn("doc2.txt", res_file["files"])
            
            # Test group by sub-corpus column (genre)
            res_genre = calculate_corpus_lexical_complexity(temp_db_path, group_by_column="genre")
            self.assertIn("news", res_genre["files"])
            self.assertIn("fiction", res_genre["files"])
            
        finally:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

if __name__ == "__main__":
    unittest.main()
