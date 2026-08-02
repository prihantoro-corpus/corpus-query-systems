import streamlit as st
import duckdb
import pandas as pd
import core.modules.overview as ov

UPOS_INFO = {
    'ADJ': {'defn': 'adjective', 'desc': 'words that modify nouns or pronouns describing properties', 'examples': ['good', 'beautiful', 'red']},
    'ADP': {'defn': 'adposition', 'desc': 'prepositions and postpositions describing relations', 'examples': ['in', 'on', 'under']},
    'ADV': {'defn': 'adverb', 'desc': 'words modifying verbs, adjectives, or other adverbs', 'examples': ['quickly', 'very', 'here']},
    'AUX': {'defn': 'auxiliary verb', 'desc': 'verbs adding grammatical info to another verb', 'examples': ['is', 'has', 'will']},
    'CCONJ': {'defn': 'coordinating conjunction', 'desc': 'conjunctions connecting words/phrases of equal rank', 'examples': ['and', 'but', 'or']},
    'DET': {'defn': 'determiner', 'desc': 'words expressing reference/quantity of a noun', 'examples': ['the', 'a', 'this']},
    'INTJ': {'defn': 'interjection', 'desc': 'words expressing emotion, exclamation, or greeting', 'examples': ['oh', 'wow', 'ouch']},
    'NOUN': {'defn': 'noun', 'desc': 'words denoting people, places, things, or concepts', 'examples': ['cat', 'house', 'beauty']},
    'NUM': {'defn': 'numeral', 'desc': 'words denoting numbers (cardinal or ordinal)', 'examples': ['one', 'three', '2026']},
    'PART': {'defn': 'particle', 'desc': 'function words associated with another word/phrase', 'examples': ['not', 'to', '\'s']},
    'PRON': {'defn': 'pronoun', 'desc': 'words substituting for nouns or noun phrases', 'examples': ['I', 'you', 'they']},
    'PROPN': {'defn': 'proper noun', 'desc': 'names of specific people, places, organizations', 'examples': ['London', 'Google', 'Sarah']},
    'PUNCT': {'defn': 'punctuation', 'desc': 'marks used to delimit sentences or clauses', 'examples': ['.', ',', '?']},
    'SCONJ': {'defn': 'subordinating conjunction', 'desc': 'conjunctions introducing subordinate clauses', 'examples': ['if', 'because', 'that']},
    'SYM': {'defn': 'symbol', 'desc': 'characters representing math, currency, or other signs', 'examples': ['$', '%', '+']},
    'VERB': {'defn': 'verb', 'desc': 'words expressing actions, occurrences, or states', 'examples': ['run', 'eat', 'sleep']},
    'X': {'defn': 'other / unknown', 'desc': 'foreign words, abbreviations, or typos', 'examples': ['etc', 'de', 'la']}
}

PTB_INFO = {
    'CC': {'defn': 'Coordinating conjunction', 'desc': 'connects words or phrases', 'examples': ['and', 'but', 'or']},
    'CD': {'defn': 'Cardinal number', 'desc': 'numerical values', 'examples': ['one', 'two', '2026']},
    'DT': {'defn': 'Determiner', 'desc': 'precedes nouns to express reference', 'examples': ['the', 'a', 'these']},
    'EX': {'defn': 'Existential there', 'desc': 'there as introductory pronoun', 'examples': ['there']},
    'FW': {'defn': 'Foreign word', 'desc': 'words from foreign languages', 'examples': ['de', 'la', 'etc']},
    'IN': {'defn': 'Preposition or subordinating conjunction', 'desc': 'relational/subordinating words', 'examples': ['in', 'on', 'because']},
    'JJ': {'defn': 'Adjective', 'desc': 'describes nouns', 'examples': ['good', 'tall', 'beautiful']},
    'JJR': {'defn': 'Adjective, comparative', 'desc': 'comparative descriptors', 'examples': ['better', 'taller', 'faster']},
    'JJS': {'defn': 'Adjective, superlative', 'desc': 'superlative descriptors', 'examples': ['best', 'tallest', 'fastest']},
    'LS': {'defn': 'List item marker', 'desc': 'markers for lists', 'examples': ['1', 'a', 'first']},
    'MD': {'defn': 'Modal', 'desc': 'modal auxiliary verbs', 'examples': ['can', 'should', 'will']},
    'NN': {'defn': 'Noun, singular or mass', 'desc': 'singular common nouns', 'examples': ['cat', 'house', 'dog']},
    'NNS': {'defn': 'Noun, plural', 'desc': 'plural common nouns', 'examples': ['cats', 'houses', 'dogs']},
    'NNP': {'defn': 'Proper noun, singular', 'desc': 'singular names', 'examples': ['London', 'Google', 'Sarah']},
    'NNPS': {'defn': 'Proper noun, plural', 'desc': 'plural names', 'examples': ['Americans', 'Capulets']},
    'PDT': {'defn': 'Predeterminer', 'desc': 'determiners preceding others', 'examples': ['all', 'both', 'half']},
    'POS': {'defn': 'Possessive ending', 'desc': 'possessive suffix', 'examples': ['\'s', '\'']},
    'PRP': {'defn': 'Personal pronoun', 'desc': 'substitutes for persons', 'examples': ['I', 'you', 'they']},
    'PRP$': {'defn': 'Possessive pronoun', 'desc': 'expresses possession', 'examples': ['my', 'your', 'their']},
    'RB': {'defn': 'Adverb', 'desc': 'modifies verbs or adjectives', 'examples': ['quickly', 'very', 'here']},
    'RBR': {'defn': 'Adverb, comparative', 'desc': 'comparative modifiers', 'examples': ['faster', 'harder', 'more']},
    'RBS': {'defn': 'Adverb, superlative', 'desc': 'superlative modifiers', 'examples': ['fastest', 'hardest', 'most']},
    'RP': {'defn': 'Particle', 'desc': 'preposition-like words acting with verbs', 'examples': ['up', 'off', 'out']},
    'SYM': {'defn': 'Symbol', 'desc': 'signs/symbols', 'examples': ['$', '%', '+']},
    'TO': {'defn': 'to', 'desc': 'infinitive marker or preposition', 'examples': ['to']},
    'UH': {'defn': 'Interjection', 'desc': 'exclamations', 'examples': ['oh', 'wow', 'ouch']},
    'VB': {'defn': 'Verb, base form', 'desc': 'infinitive or present verbs', 'examples': ['run', 'eat', 'sleep']},
    'VBD': {'defn': 'Verb, past tense', 'desc': 'past tense action verbs', 'examples': ['ran', 'ate', 'slept']},
    'VBG': {'defn': 'Verb, gerund or present participle', 'desc': '-ing verbs', 'examples': ['running', 'eating', 'sleeping']},
    'VBN': {'defn': 'Verb, past participle', 'desc': 'past participle verbs', 'examples': ['run', 'eaten', 'slept']},
    'VBP': {'defn': 'Verb, non-3rd person singular present', 'desc': 'present verbs with I/you/we/they', 'examples': ['run', 'eat', 'sleep']},
    'VBZ': {'defn': 'Verb, 3rd person singular present', 'desc': 'present verbs with he/she/it', 'examples': ['runs', 'eats', 'sleeps']},
    'WDT': {'defn': 'Wh-determiner', 'desc': 'wh-question determiners', 'examples': ['which', 'what', 'that']},
    'WP': {'defn': 'Wh-pronoun', 'desc': 'wh-question pronouns', 'examples': ['who', 'what', 'whom']},
    'WP$': {'defn': 'Possessive wh-pronoun', 'desc': 'possessive wh-pronouns', 'examples': ['whose']},
    'WRB': {'defn': 'Wh-adverb', 'desc': 'wh-question adverbs', 'examples': ['where', 'when', 'how']}
}

@st.cache_data(show_spinner=False)
def explain_pos_tag_via_spacy(tag):
    """
    Explain a POS tag using custom glossary (Indonesian BPPT, lexical/auxiliary verb extensions) or spaCy glossary.
    """
    if not tag:
        return ""
        
    tag_upper = tag.upper()
    
    CUSTOM_GLOSSARY = {
        # Indonesian BPPT tags
        "NEG": "negation (kata penyangkalan, e.g., tidak, bukan)",
        "SC": "subordinating conjunction (kata hubung anak kalimat, e.g., karena, jika)",
        "CC": "coordinating conjunction (kata hubung setara, e.g., dan, tetapi)",
        "CD": "cardinal / numeral (kata bilangan, e.g., satu, dua)",
        "DT": "determiner (kata sandang/penentu, e.g., itu, sebuah)",
        "FW": "foreign word (kata asing)",
        "IN": "preposition (kata depan, e.g., di, ke, dari)",
        "JJ": "adjective (kata sifat, e.g., bagus, besar)",
        "MD": "modal / auxiliary (kata bantu, e.g., akan, dapat)",
        "NN": "common noun (kata benda am)",
        "NNP": "proper noun (kata benda khas)",
        "PRP": "personal pronoun (kata ganti nama diri, e.g., saya, kamu)",
        "RB": "adverb (kata keterangan, e.g., sangat, cepat)",
        "RP": "particle (partikel)",
        "UH": "interjection (kata seru, e.g., wah, aduh)",
        "VB": "verb (kata kerja)",
        "Z": "punctuation (tanda baca)",
        "FOC": "focus particle (partikel penegas, e.g., -lah, -kah)",
        "KUA": "quantifier (kata penentu jumlah)",
        "VSD": "intransitive verb (kata kerja intransitif)",
        "VBT": "transitive verb (kata kerja transitif)",
        "NSD": "common noun (kata benda am)",
        "NSM": "proper noun (kata benda khas)",
        
        # Lexical Verb tags (e.g. CLAWS / Treebank extensions)
        "VV": "verb, lexical, base form (e.g., take, run)",
        "VVB": "verb, lexical, base form (e.g., take, run)",
        "VVD": "verb, lexical, past tense (e.g., took, ran)",
        "VVG": "verb, lexical, gerund or present participle (e.g., taking, running)",
        "VVN": "verb, lexical, past participle (e.g., taken, run)",
        "VVP": "verb, lexical, present tense, non-3rd person singular (e.g., take, run)",
        "VVZ": "verb, lexical, present tense, 3rd person singular (e.g., takes, runs)",
        
        # Auxiliary Be verb tags
        "VBD": "verb, auxiliary/be, past tense (e.g., was, were)",
        "VBG": "verb, auxiliary/be, gerund or present participle (e.g., being)",
        "VBN": "verb, auxiliary/be, past participle (e.g., been)",
        "VBP": "verb, auxiliary/be, present tense, non-3rd person singular (e.g., am, are)",
        "VBZ": "verb, auxiliary/be, present tense, 3rd person singular (e.g., is)",
        
        # Auxiliary Have verb tags
        "VH": "verb, auxiliary/have, base form (e.g., have)",
        "VHD": "verb, auxiliary/have, past tense (e.g., had)",
        "VHG": "verb, auxiliary/have, gerund or present participle (e.g., having)",
        "VHN": "verb, auxiliary/have, past participle (e.g., had)",
        "VHP": "verb, auxiliary/have, present tense, non-3rd person singular (e.g., have)",
        "VHZ": "verb, auxiliary/have, present tense, 3rd person singular (e.g., has)",
        
        # Auxiliary Do verb tags
        "VD": "verb, auxiliary/do, base form (e.g., do)",
        "VDD": "verb, auxiliary/do, past tense (e.g., did)",
        "VDG": "verb, auxiliary/do, gerund or present participle (e.g., doing)",
        "VDN": "verb, auxiliary/do, past participle (e.g., done)",
        "VDP": "verb, auxiliary/do, present tense, non-3rd person singular (e.g., do)",
        "VDZ": "verb, auxiliary/do, present tense, 3rd person singular (e.g., does)"
    }
    if tag_upper in CUSTOM_GLOSSARY:
        return CUSTOM_GLOSSARY[tag_upper]
        
    # Check spacy glossary
    try:
        import spacy
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = spacy.explain(tag)
        if res:
            return res
    except:
        pass
        
    return ""

@st.cache_data(show_spinner=False)
def get_pos_tag_examples(db_path, tag):
    """
    Query 3 distinct random words for this tag from the corpus.
    Caches the results in a 'pos_examples' table within the database for performance.
    """
    if not db_path:
        return ""
        
    # 1. Try reading from pos_examples table first
    con = duckdb.connect(db_path)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if 'pos_examples' in tables:
            res = con.execute("SELECT examples FROM pos_examples WHERE tag = ?", [tag]).fetchone()
            if res:
                return res[0]
    except Exception:
        pass
    finally:
        con.close()
        
    # 2. Not found, compute it
    examples_str = ""
    con = duckdb.connect(db_path)
    try:
        res = con.execute("""
            SELECT DISTINCT token 
            FROM corpus 
            WHERE pos = ? 
              AND NOT regexp_matches(token, '^[[:punct:]\\s]+$')
            LIMIT 10
        """, [tag]).fetchall()
        
        import random
        words = [r[0] for r in res if r[0]]
        if len(words) > 3:
            random.seed(hash(tag))
            words = random.sample(words, 3)
        examples_str = ", ".join(words)
    except Exception:
        try:
            res = con.execute("SELECT DISTINCT token FROM corpus WHERE pos = ? LIMIT 3", [tag]).fetchall()
            examples_str = ", ".join([r[0] for r in res if r[0]])
        except:
            examples_str = ""
    finally:
        con.close()
        
    # 3. Cache it in the database
    if examples_str:
        try:
            con_write = duckdb.connect(db_path, read_only=False)
            try:
                con_write.execute("CREATE TABLE IF NOT EXISTS pos_examples (tag VARCHAR PRIMARY KEY, examples VARCHAR)")
                con_write.execute("DELETE FROM pos_examples WHERE tag = ?", [tag])
                con_write.execute("INSERT INTO pos_examples VALUES (?, ?)", [tag, examples_str])
            finally:
                con_write.close()
        except Exception:
            pass
            
    return examples_str

def infer_tagger_and_tagset(db_path):
    """
    Infers the POS tagger and tagset used based on corpus metadata and unique tags.
    """
    if not db_path:
        return "Unknown", "Unknown"
    
    tagger = "Unknown"
    tagset = "Unknown"
    
    con = duckdb.connect(db_path)
    try:
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if 'corpus_metadata' in tables:
            res = con.execute("SELECT value FROM corpus_metadata WHERE key='tagger'").fetchone()
            if res:
                tagger = res[0]
            res_ts = con.execute("SELECT value FROM corpus_metadata WHERE key='tagset'").fetchone()
            if res_ts:
                tagset = res_ts[0]
                
        # If not determined, infer from the unique tags in corpus
        if tagset == "Unknown" or tagger == "Unknown":
            # Check POS column existence
            cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
            cols = [c[1] for c in cols_info]
            if 'pos' not in cols:
                return "None", "No POS tags in corpus"
                
            res_tags = con.execute("SELECT DISTINCT pos FROM corpus WHERE pos NOT IN ('##', '###', 'O', '') AND pos NOT LIKE '##%' LIMIT 50").fetchall()
            tags = {r[0] for r in res_tags if r[0]}
            
            upos_tags = {'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'}
            ptb_tags = {'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'}
            
            if not tags:
                tagset = "None / Empty"
                tagger = "None / Empty"
            elif tags.issubset(upos_tags) or len(tags.intersection(upos_tags)) / max(len(tags), 1) > 0.7:
                tagset = "Universal Dependencies (UPOS)"
                tagger = "spaCy (with Stanza fallback)"
            elif any(t.startswith('NS') or t in ('VSD', 'NEG', 'KUA', 'FOC') for t in tags):
                tagset = "ID-BPPT Indonesian Tagset"
                tagger = "ID-BPPT Tagger"
            elif len(tags.intersection(ptb_tags)) / max(len(tags), 1) > 0.5:
                tagset = "Penn Treebank (PTB)"
                tagger = "Pre-tagged / spaCy"
            else:
                tagset = "Custom / Other"
                tagger = "Pre-tagged"
    except Exception as e:
        print(f"Error inferring tagger: {e}")
    finally:
        con.close()
        
    return tagger, tagset

def render_pos_help_button(db_path, key_suffix=""):
    """
    Renders a help popover button that lists POS tags, definitions, and corpus examples.
    """
    if not db_path:
        return
        
    tagger, tagset = infer_tagger_and_tagset(db_path)
    tags = ov.get_unique_pos_tags(db_path)
    
    if not tags:
        return
        
    with st.popover("❓ POS Tag Guide", use_container_width=False):
        st.markdown("### 🏷️ Part-of-Speech Tag Guide")
        st.markdown(f"🤖 **Pipeline Tagger:** `{tagger}`  \n🏷️ **Tagset Scheme:** `{tagset}`")
        
        current_defs = ov.get_pos_definitions(db_path)
        tagset_lower = tagset.lower()
        if "upos" in tagset_lower or "universal" in tagset_lower:
            standard_info = UPOS_INFO
        elif "penn" in tagset_lower or "ptb" in tagset_lower:
            standard_info = PTB_INFO
        else:
            standard_info = {}
            
        data_rows = []
        for t in tags:
            defn = current_defs.get(t, "")
            if not defn:
                if t in standard_info:
                    defn = f"{standard_info[t]['defn']} ({standard_info[t]['desc']})"
                else:
                    spacy_defn = explain_pos_tag_via_spacy(t)
                    if spacy_defn:
                        defn = spacy_defn
                    else:
                        defn = "No definition available."
                
            examples = get_pos_tag_examples(db_path, t)
            data_rows.append({
                "Tag": t,
                "Definition": defn,
                "Examples (from corpus)": examples if examples else "None"
            })
            
        st.dataframe(
            pd.DataFrame(data_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tag": st.column_config.TextColumn("Tag", width=120),
                "Definition": st.column_config.TextColumn("Definition", width=350),
                "Examples (from corpus)": st.column_config.TextColumn("Examples (from corpus)", width=250)
            }
        )

NER_INFO = {
    'PERSON': {'defn': 'Person', 'desc': 'People, including fictional.', 'examples': ['Albert Einstein', 'Harry Potter']},
    'NORP': {'defn': 'Nationalities or religious/political groups', 'desc': 'Nationalities, religious or political groups.', 'examples': ['Americans', 'Democrats']},
    'FAC': {'defn': 'Facility', 'desc': 'Buildings, airports, highways, bridges, etc.', 'examples': ['Golden Gate Bridge', 'JFK Airport']},
    'ORG': {'defn': 'Organization', 'desc': 'Companies, agencies, institutions, etc.', 'examples': ['Apple', 'United Nations']},
    'GPE': {'defn': 'Geopolitical Entity', 'desc': 'Countries, cities, states.', 'examples': ['New York', 'Japan']},
    'LOC': {'defn': 'Location', 'desc': 'Non-GPE locations, mountain ranges, bodies of water.', 'examples': ['Mount Everest', 'Pacific Ocean']},
    'PRODUCT': {'defn': 'Product', 'desc': 'Objects, vehicles, foods, etc. (Not services.)', 'examples': ['iPhone', 'Honda Civic']},
    'EVENT': {'defn': 'Event', 'desc': 'Named hurricanes, battles, wars, sports events, etc.', 'examples': ['World War II', 'Olympics']},
    'WORK_OF_ART': {'defn': 'Work of Art', 'desc': 'Titles of books, songs, etc.', 'examples': ['Mona Lisa', 'The Beatles']},
    'LAW': {'defn': 'Law', 'desc': 'Named documents made into laws.', 'examples': ['Constitution', 'Bill of Rights']},
    'LANGUAGE': {'defn': 'Language', 'desc': 'Any named language.', 'examples': ['English', 'Spanish']},
    'DATE': {'defn': 'Date', 'desc': 'Absolute or relative dates or periods.', 'examples': ['January', '2026']},
    'TIME': {'defn': 'Time', 'desc': 'Times smaller than a day.', 'examples': ['morning', '8:00 AM']},
    'PERCENT': {'defn': 'Percent', 'desc': 'Percentage, including "%".', 'examples': ['20%', 'fifty percent']},
    'MONEY': {'defn': 'Money', 'desc': 'Monetary values, including unit.', 'examples': ['$100', '50 euros']},
    'QUANTITY': {'defn': 'Quantity', 'desc': 'Measurements, as of weight or distance.', 'examples': ['10 kg', '5 miles']},
    'ORDINAL': {'defn': 'Ordinal Number', 'desc': '"first", "second", etc.', 'examples': ['first', '2nd']},
    'CARDINAL': {'defn': 'Cardinal Number', 'desc': 'Numerals that do not fall under another type.', 'examples': ['two', '1000']}
}

DEP_INFO = {
    'acl': {'defn': 'clausal modifier of noun', 'desc': 'Finite or non-finite clause that modifies a noun.', 'examples': ['the man *who is singing*']},
    'advcl': {'defn': 'adverbial clause modifier', 'desc': 'A clause that modifies a verb or other predicate as an adjunct.', 'examples': ['he ran *because he was scared*']},
    'advmod': {'defn': 'adverbial modifier', 'desc': 'A (non-clausal) adverb or adverbial phrase that serves to modify a predicate.', 'examples': ['he ran *quickly*']},
    'amod': {'defn': 'adjectival modifier', 'desc': 'An adjectival phrase that modifies a noun.', 'examples': ['the *red* car']},
    'appos': {'defn': 'appositional modifier', 'desc': 'A noun phrase that serves to define or modify another noun phrase.', 'examples': ['Sam, my *brother*']},
    'aux': {'defn': 'auxiliary', 'desc': 'A non-main verb of the clause.', 'examples': ['he *is* running']},
    'case': {'defn': 'case marking', 'desc': 'Prepositions, postpositions, and other case markers.', 'examples': ['*in* the house']},
    'cc': {'defn': 'coordinating conjunction', 'desc': 'A conjunction that links two coordinated elements.', 'examples': ['apples *and* oranges']},
    'ccomp': {'defn': 'clausal complement', 'desc': 'A dependent clause which is a core argument.', 'examples': ['he says *that he is happy*']},
    'compound': {'defn': 'compound', 'desc': 'A compound noun or multi-word expression.', 'examples': ['*apple* juice']},
    'conj': {'defn': 'conjunct', 'desc': 'The rightmost element of a coordinated structure.', 'examples': ['apples and *oranges*']},
    'cop': {'defn': 'copula', 'desc': 'A linking verb.', 'examples': ['he *is* happy']},
    'csubj': {'defn': 'clausal subject', 'desc': 'A clausal syntactic subject of a clause.', 'examples': ['*what she said* makes sense']},
    'dep': {'defn': 'unspecified dependency', 'desc': 'A dependency that cannot be assigned a more specific label.', 'examples': []},
    'det': {'defn': 'determiner', 'desc': 'A word that expresses reference of a noun.', 'examples': ['*the* car', '*a* dog']},
    'fixed': {'defn': 'multi-word expression', 'desc': 'Grammaticalized multi-word expression.', 'examples': ['*in spite of*']},
    'flat': {'defn': 'flat multi-word expression', 'desc': 'Names and other multi-word expressions without internal syntactic structure.', 'examples': ['*New York*']},
    'goeswith': {'defn': 'goes with', 'desc': 'Links parts of a word that are separated in text.', 'examples': []},
    'iobj': {'defn': 'indirect object', 'desc': 'The recipient or beneficiary of an action.', 'examples': ['give *him* the book']},
    'list': {'defn': 'list', 'desc': 'Chains of comparable items.', 'examples': []},
    'mark': {'defn': 'marker', 'desc': 'A word introducing a finite clause subordinate to another clause.', 'examples': ['he ran *because* he was scared']},
    'nmod': {'defn': 'nominal modifier', 'desc': 'A noun phrase that serves to modify another noun phrase.', 'examples': ['the door of the *house*']},
    'nsubj': {'defn': 'nominal subject', 'desc': 'A noun phrase which is the syntactic subject of a clause.', 'examples': ['*the car* is red']},
    'nsubjpass': {'defn': 'nominal subject (passive)', 'desc': 'A noun phrase which is the syntactic subject of a passive clause.', 'examples': ['*the car* was stolen']},
    'nummod': {'defn': 'numeric modifier', 'desc': 'A number that serves to modify the meaning of a noun.', 'examples': ['*three* cars']},
    'obj': {'defn': 'object', 'desc': 'The direct object of a verb.', 'examples': ['eat *an apple*']},
    'dobj': {'defn': 'direct object', 'desc': 'The direct object of a verb.', 'examples': ['eat *an apple*']},
    'obl': {'defn': 'oblique nominal', 'desc': 'A non-core nominal argument.', 'examples': ['give the book *to him*']},
    'orphan': {'defn': 'orphan', 'desc': 'Used for elliptical constructions.', 'examples': []},
    'parataxis': {'defn': 'parataxis', 'desc': 'A clause placed side by side with another clause without coordination or subordination.', 'examples': ['he came, *he saw*, he conquered']},
    'punct': {'defn': 'punctuation', 'desc': 'Punctuation marks.', 'examples': ['.', ',']},
    'reparandum': {'defn': 'overridden disfluency', 'desc': 'Speech repair.', 'examples': []},
    'root': {'defn': 'root', 'desc': 'The root of the sentence, typically the main verb.', 'examples': []},
    'vocative': {'defn': 'vocative', 'desc': 'A name or noun phrase used to address someone.', 'examples': ['*John*, come here']},
    'xcomp': {'defn': 'open clausal complement', 'desc': 'A clausal complement without its own subject.', 'examples': ['he wants *to sleep*']}
}

SENTIMENT_INFO = {
    'Positive': {'defn': 'Positive Sentiment', 'desc': 'The text segment expresses a favorable, supportive, or optimistic polarity.', 'examples': []},
    'Neutral': {'defn': 'Neutral Sentiment', 'desc': 'The text segment is objective, factual, or lacks strong emotional polarity.', 'examples': []},
    'Negative': {'defn': 'Negative Sentiment', 'desc': 'The text segment expresses an unfavorable, critical, or pessimistic polarity.', 'examples': []}
}

@st.cache_data(show_spinner=False)
def get_tag_examples(db_path, tag, column):
    if not db_path:
        return ""
    examples_str = ""
    import duckdb
    con = duckdb.connect(db_path)
    try:
        res = con.execute(f"SELECT DISTINCT token FROM corpus WHERE {column} = ? AND NOT regexp_matches(token, '^[[:punct:]\\s]+$') LIMIT 10", [tag]).fetchall()
        import random
        words = [r[0] for r in res if r[0]]
        if len(words) > 3:
            random.seed(hash(tag))
            words = random.sample(words, 3)
        examples_str = ", ".join(words)
    except Exception:
        pass
    finally:
        con.close()
    return examples_str

def check_available_annotations(db_path):
    available = ["Part-of-Speech"]
    try:
        import duckdb
        with duckdb.connect(db_path, read_only=True) as con:
            cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
            cols = [c[1].lower() for c in cols_info]
            if "ent_type" in cols or "in_ner_start" in cols:
                available.append("Named Entity Recognition (NER)")
            if "dep_rel" in cols:
                available.append("Dependency Parsing")
            if "sentiment" in cols:
                available.append("Sentiment Analysis")
    except:
        pass
    return available

def render_annotation_help_button(db_path, key_suffix=""):
    if not db_path:
        return
        
    layers = check_available_annotations(db_path)
    
    with st.popover("❓ Annotation Tags Guide", use_container_width=False):
        st.markdown("### 🏷️ Annotation Tags Guide")
        
        selected_layer = st.radio("Select Annotation Layer", layers, key=f"anno_layer_{key_suffix}", horizontal=True)
        
        if selected_layer == "Part-of-Speech":
            tagger, tagset = infer_tagger_and_tagset(db_path)
            tags = ov.get_unique_pos_tags(db_path)
            if not tags:
                st.info("No POS tags detected.")
                return
            st.markdown(f"🤖 **Pipeline Tagger:** `{tagger}`  \n🏷️ **Tagset Scheme:** `{tagset}`")
            current_defs = ov.get_pos_definitions(db_path)
            tagset_lower = tagset.lower()
            if "upos" in tagset_lower or "universal" in tagset_lower:
                standard_info = UPOS_INFO
            elif "penn" in tagset_lower or "ptb" in tagset_lower:
                standard_info = PTB_INFO
            else:
                standard_info = {}
            
            data_rows = []
            for t in tags:
                defn = current_defs.get(t, "")
                if not defn:
                    if t in standard_info:
                        defn = f"{standard_info[t]['defn']} ({standard_info[t]['desc']})"
                    else:
                        spacy_defn = explain_pos_tag_via_spacy(t)
                        if spacy_defn:
                            defn = spacy_defn
                        else:
                            defn = "No definition available."
                
                examples = get_pos_tag_examples(db_path, t)
                data_rows.append({
                    "Tag": t,
                    "Definition": defn,
                    "Examples": examples if examples else "None"
                })
        
        elif selected_layer == "Named Entity Recognition (NER)":
            column = "ent_type"
            import duckdb
            try:
                with duckdb.connect(db_path, read_only=True) as con:
                    res_tags = con.execute("SELECT DISTINCT ent_type FROM corpus WHERE ent_type IS NOT NULL AND ent_type != '' AND ent_type != 'O'").fetchall()
                    tags = [r[0] for r in res_tags if r[0]]
            except:
                tags = []
            if not tags:
                st.info("No NER tags detected.")
                return
            
            data_rows = []
            for t in tags:
                t_clean = t.replace('B-', '').replace('I-', '')
                if t_clean in NER_INFO:
                    defn = f"{NER_INFO[t_clean]['defn']} ({NER_INFO[t_clean]['desc']})"
                else:
                    defn = "No definition available."
                examples = get_tag_examples(db_path, t, "ent_type")
                data_rows.append({
                    "Tag": t,
                    "Definition": defn,
                    "Examples": examples if examples else "None"
                })
        
        elif selected_layer == "Dependency Parsing":
            column = "dep_rel"
            import duckdb
            try:
                with duckdb.connect(db_path, read_only=True) as con:
                    res_tags = con.execute("SELECT DISTINCT dep_rel FROM corpus WHERE dep_rel IS NOT NULL AND dep_rel != ''").fetchall()
                    tags = [r[0] for r in res_tags if r[0]]
            except:
                tags = []
            if not tags:
                st.info("No Dependency tags detected.")
                return
            
            data_rows = []
            for t in tags:
                if t in DEP_INFO:
                    defn = f"{DEP_INFO[t]['defn']} ({DEP_INFO[t]['desc']})"
                else:
                    defn = "No definition available."
                examples = get_tag_examples(db_path, t, "dep_rel")
                data_rows.append({
                    "Tag": t,
                    "Definition": defn,
                    "Examples": examples if examples else "None"
                })
                
        elif selected_layer == "Sentiment Analysis":
            column = "sentiment"
            import duckdb
            try:
                with duckdb.connect(db_path, read_only=True) as con:
                    res_tags = con.execute("SELECT DISTINCT sentiment FROM corpus WHERE sentiment IS NOT NULL AND sentiment != ''").fetchall()
                    tags = [r[0] for r in res_tags if r[0]]
            except:
                tags = []
            if not tags:
                st.info("No Sentiment tags detected.")
                return
            
            data_rows = []
            for t in tags:
                t_clean = t.capitalize() if isinstance(t, str) else t
                if t_clean in SENTIMENT_INFO:
                    defn = f"{SENTIMENT_INFO[t_clean]['defn']} ({SENTIMENT_INFO[t_clean]['desc']})"
                else:
                    defn = "No definition available."
                data_rows.append({
                    "Tag": t,
                    "Definition": defn,
                    "Examples": "N/A (Document-level metric)"
                })
                
        if data_rows:
            st.dataframe(
                pd.DataFrame(data_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Tag": st.column_config.TextColumn("Tag", width=120),
                    "Definition": st.column_config.TextColumn("Definition", width=350),
                    "Examples": st.column_config.TextColumn("Examples", width=250)
                }
            )
