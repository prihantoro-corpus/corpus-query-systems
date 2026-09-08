=== Rule-Based Custom Tagger Directory ===

Rule Files:
  1. Lexicon.txt
     TreeTagger format lexicon. Multi-word units (e.g. White House) and ambiguity (e.g. saw NOUN saw VERB see) are supported.
  2. Lemma.txt
     Code to lemma mapping file (e.g. V013 fly, V014 see).
  3. Word-formation.txt
     Inflection and derivation rules (e.g. V013 VERB <D1>ing, V013 VERB <D1>ies).
  4. Word-form.txt
     Synthesized full word form dictionary created from Lemma.txt and Word-formation.txt.
  5. Guesser-regex.txt
     Regex/suffix guesser rules (e.g. VERB ing, NOUN tion, ADJ al).
  6. Guesser1.txt
     One tag guesser default fallback (e.g. NOUN).
  7. rule_based_tagger_model.json
     Serialized tagger model JSON containing all files and priority order settings.

Sample Input & Output Files:
  8. sample_corpus.txt
     Sample input text for testing the rule-based tagger.
  9. sample_tagged_output.txt
     Vertical TreeTagger-style output of tagged corpus tokens.
 10. sample_step_by_step_output.txt
     Detailed trace of how each rule file in priority order tags each token.
