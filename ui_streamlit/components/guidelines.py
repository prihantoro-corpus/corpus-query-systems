import streamlit as st
from ui_streamlit.state_manager import get_state, set_state

GUIDELINES = {
    "Overview": {
        "XML Structure": """
            ### 📖 XML Structure Guide
            * **Structure & Attributes**: View the XML tag hierarchy and document/token attributes extracted from the source files.
            * **Raw Python Data**: Explores the raw parsed dictionary representation for troubleshooting.
            * **Database Diagnostics**: Checks the active DuckDB table schema and detected metadata columns.
        """,
        "Sub-corpus Stats": """
            ### 📖 Sub-Corpus Statistics Guide
            * **Metadata Breakdown**: View token, type, and document distributions partitioned by metadata attributes (e.g. year, genre, author).
            * **TTR by Sub-corpus**: Analyze type/token ratio variations across sub-corpora to compare lexical diversity.
        """,
        "Top Frequencies": """
            ### 📖 Top Frequencies Guide
            * **Vocabulary Ranking**: Lists the most frequent tokens, lemmas, and POS tags in the corpus.
            * **PMW Frequency**: Displays normalized relative frequencies (Parts Per Million / PMW) for standard comparison.
            * **Excel Export**: Download the full frequency table as a spreadsheet.
        """,
        "Unique POS Tags": """
            ### 📖 Unique POS Tags Guide
            * **Grammatical Distribution**: Lists all unique Part-of-Speech tags detected in the corpus.
            * **Language-Specific Mapping**: Explains what each tag stands for (e.g., *NN* for noun, *VB* for verb) under the confirmed language standard.
        """,
        "Word Cloud": """
            ### 📖 Word Cloud Guide
            * **Visual Vocabulary**: Displays a word cloud of the top frequent terms in the corpus.
            * **Parameters**: Adjust size, colors, and maximum words to display. Larger font sizes correspond to higher frequencies.
        """,
        "Metadata Annotation": """
            ### 📖 Metadata Annotation Guide
            * **Custom Tagging**: Add or edit document-level attributes directly within the system interface.
            * **Import/Export**: Export metadata profiles to local files or load external annotations.
        """,
        "🏷️ Sentiment & Topic Analysis": """
            ### 📖 Sentiment & Topic Analysis Guide
            * **VADER Sentiment**: Computes positive, neutral, and negative sentiment distribution across your texts.
            * **Topic Modeling**: Automatically extracts topic clusters using BERTTopic (semantic embeddings) or Keyword-Weighted algorithms.
        """,
        "🏷️ Named Entity Recognition (NER)": """
            ### 📖 Named Entity Recognition (NER) Guide
            * **Entity Classification**: Automatically extracts and labels proper nouns into categories (e.g. *Person*, *Organization*, *Location*, *Date*).
            * **Frequency Rankings**: Lists the most prominent entities found across the corpus.
        """,
        "📖 Reading Ease": """
            ### 📖 Reading Ease Guide
            * **Readability Metrics**: Evaluates texts using standard indexes (Flesch Reading Ease, Flesch-Kincaid Grade Level, LIX, RIX).
            * **Text Complexity**: Evaluates average sentence lengths, syllable counts, and chunk readability to assess difficulty.
        """
    },
    "Concordance": """
        ### 📖 Concordance (KWIC) User Guide
        
        A concordance displays occurrences of a query word (the **node word**) in its immediate context. Here is how to use the Concordance module:
        
        #### 1. Node Word Input
        You can search for direct words, wildcards, lemmas, or specific grammatical patterns:
        * **Direct Token**: Type a literal word (e.g., `run` or `beautiful`).
        * **Wildcard (`*`)**: Use `*` to match prefix, suffix, or parts of words.
          * Example: `run*` matches *run*, *runs*, *running*, *runner*.
          * Example: `*ing` matches *sing*, *playing*, *jumping*.
        * **Lemma Unit**: Wrap the base form in square brackets to match all inflected forms.
          * Example: `[run]` matches *run*, *ran*, *running*, *runs*.
        * **POS Restricted Tokens**: Search for words with specific Part-of-Speech tags.
          * Example: `light_V*` matches *light* when used as a verb.
          * Example: `_NN` matches any singular noun.
        * **XML Tags**: Match structural XML tags directly.
          * Example: `<PN type="human">`
          
        #### 2. Context Window & Max Lines
        * **Context Window**: Adjusts how many words are retrieved to the left and right of the node word (e.g., 5 words on each side).
        * **Max Lines**: Limits the maximum number of matching rows to return (e.g., 100 lines), ensuring fast responsiveness.
        
        #### 3. Filtering by Collocate (Regex)
        * Apply a Regular Expression filter to narrow down results to only those where a specific pattern exists in the surrounding context.
        * Example: `\\b(very|extremely)\\b` filters for lines containing either "very" or "extremely" near the node word.
        * Example: `not` filters for lines containing the word "not".
        
        #### 4. Display Checkboxes (POS, Lemma, Metadata)
        * **Show POS**: Appends Part-of-Speech tags to all words in the context windows.
        * **Show Lemma**: Displays the base dictionary form for all words in the context windows.
        * **Show Metadata**: Displays document/corpus metadata (such as Text ID, source, or subcorpus classification) in a column on the left.
        
        #### 5. Query Triggering
        * Once you are satisfied with these controls, click the **Generate Concordance Lines** (or **Generate Concordance**) button to run the query and generate the concordance lines.
        
        #### 6. Clustering by Sub-corpora (Advanced)
        * To cluster the results of concordance lines:
          1. In the **Advanced** tab, first restrict searches using the **XML Restriction Filters** based on sub-corpora present in the corpus.
          2. Generate your initial concordance lines.
          3. Select your categorical metadata filters in the restricted search.
          4. Click **🧩 Cluster Mode** to group and sample results based on the selected sub-corpora.
          * **Example**: You can compare and analyze the presence of concordance lines in different combinations, such as *negative sentiment* vs. *positive sentiment*, or *negative sentiments in the economy section* vs. *negative sentiments in the sports section*.

        #### 7. Interactive Annotation Mode
        * **Annotate Lines**: Toggle **✍️ Annotation Mode** to start annotating the concordance lines. You can tag occurrences with your own invented custom attributes and values.
        * **Save & Resume**: You do not have to complete your annotations all at once. Click **💾 Save Annotation Progress** to save your annotations to a JSON file on your machine.
        * **Upload & Restore**: When you are ready to resume, click **📁 Continue Annotation** and upload your saved JSON file to instantly restore your annotation progress.
        * **Index & Retrieve**: Once you are happy with the labels, click **🏛️ Apply to Session** to index and save these annotations directly into the corpus database. They are then fully retrieved and searchable across other views (like Overview and Restricted Search).
    """,
    "N-Gram": """
        ### 📖 N-Gram Analysis User Guide
        
        An N-Gram is a contiguous sequence of *N* items (words, lemmas, or POS tags) from a given text. This module helps you find and analyze frequent word patterns in your corpus:
        
        #### 1. N-Gram Size & General Settings
        * **N-Gram Size (N)**: Use the slider to select the length of the sequences (e.g., 2 for Bigrams, 3 for Trigrams, etc.).
        * **Skip Punctuation**: Excludes punctuation marks and special symbols from the sequences.
        * **Output Basis**: Select the default base representation of the n-gram elements:
          * **Token**: Matches raw, literal words (e.g., *went*, *running*).
          * **Lemma**: Matches base dictionary forms (e.g., *go*, *run*).
          * **POS Tag**: Matches part-of-speech categories (e.g., *VBD*, *VBG*).
        
        #### 2. Advanced Positional Filters
        In **Advanced** mode, you can customize the **Basis** and specify **Filters** for *each position* in the N-gram sequence separately:
        * **Wildcards**: Use `*`, `%`, or `_` to match partial words (e.g. `inter*` matches *international*, *internet*).
        * **POS Tag Suffixes**: Match specific parts of speech by appending `_TAG` (e.g., `_NN` for nouns).
        * **Lemma Override**: Search by lemma in a token-based query by wrapping the lemma in brackets (e.g., `[be]` matches *am*, *is*, *are*, *was*, *were*).
        * **Negation**: Prefix a word with a minus sign (`-`) to exclude it from that position (e.g. `-the` matches any word except *the*).
        
        #### 3. XML Restriction Filters
        * Filter the source data before generating n-grams based on document-level metadata (sub-corpora attributes such as *author*, *publication year*, or *sentiment*).
        
        #### 4. Query Triggering
        * After choosing settings or positioning filters, click **Generate N-Grams** (or **Generate Comparison N-Grams** in Comparison Mode) to run the analysis.
        
        #### 5. Results & Interpretation
        * **Metrics**: Results display the Absolute Frequency, Relative Frequency (per Million Words / PMW), Zipf Scores, and Zipf Law Frequency Bands.
        * **Excel Export**: Download the full list of n-gram patterns as an Excel spreadsheet.
        * **Interpret with AI**: Submit the top patterns to the LLM to get a detailed semantic analysis.
    """,
    "Collocation": """
        ### 📖 Collocation Analysis User Guide
        
        Collocations are pairs or groups of words that co-occur more frequently than would be expected by chance.
        
        #### 1. Search Settings
        * **Node Word**: Input the target word around which to find collocates.
        * **Association Measure**: Select the formula used to calculate collocation strength:
          * **Log-Likelihood (LL)**: Best for high-frequency patterns.
          * **Log-Dice**: Reflects the exclusive co-occurrence (independent of corpus size).
          * **Mutual Information (MI)**: Emphasizes strongly bound, rare terms.
          * **Dice Coefficient**: Evaluates overlap ratio.
        * **Context Window**: Determine the span of words surrounding the node word to search (e.g., 5 words to the left/right).
        * **Show all collocates in concordance**: Check this box to retrieve and display **all** matching concordance lines for each collocate instead of a single sample instance.
          * ⚠️ *Warning: This will query and show all occurrences in the concordance and may take significant time to load.*
        
        #### 2. Advanced Filters & XML Restrictions
        * Apply positional filters (Tokens/Lemmas/POS) and XML restrictions to analyze collocations in specific sub-corpora.
        
        #### 3. Collocation Patterns (Optional)
        Cluster collocates dynamically using grammar patterns defined line-by-line (`label : pattern`).
        * **Syntax Symbols**:
          * `#` : Represents the **node word**.
          * `<...>` : Represents the **collocate**.
          * `*` : Optional token (0 or 1 words).
          * `+` : Required token (exactly 1 word).
          * `_TAG` : POS tag constraint (e.g. `_VB`).
          * `[lemma]` : Lemma constraint (e.g. `[be]`).
        * **Bracketed Collocate Filters (<...>)**:
          * Filter the collocate placeholder `<...>` using POS, lemma, or literal tokens with OR (`|`) semantics inside.
          * Example: `<_VB>` (verbs), `<are|is|am>` (tokens), `<[be]>` (lemma `be`), `<_NN|_PP|_NNP>` (nouns or pronouns).
        * **Pattern Unions (`|`)**:
          * Specify multiple structural options/patterns on a single label line using the outer union symbol `|`.
          * Example: `agent of passive : # *ed by <_NN|_PP|_NNP> | # *ed by the <_NN|_PP|_NNP>` (retrieves both *"killed by Jack"* and *"killed by the enemy"*).
        
        #### 4. Interactive Visualization
        * **Charts & Networks**: Dynamically view collocation scores plotted on graphs. Use the PyVis Network Graph to explore associative paths visually.
    """,
    "Dictionary": """
        ### 📖 Dictionary User Guide
        
        The Dictionary module provides built-in dictionary lookups, definition tracking, and thesaurus synonyms:
        
        #### 1. Definition Lookup
        * Type any word to fetch standard semantic definitions, phonetic transcriptions, grammatical categories, and usage examples.
        * Uses your confirmed corpus language settings to route requests to the correct language database.
        
        #### 2. Thesaurus & Synonyms
        * Find synonyms, antonyms, and related words to explore lexical variations and vocabulary alternatives.
    """,
    "Word Profiler": """
        ### 📖 Word Profiler User Guide
        
        The Word Profiler builds a multi-dimensional lexical profile for a target word:
        
        #### 1. Lexical Diagnostics
        * Displays the word's absolute frequency, relative frequency (PMW), and distribution rank.
        * Plots part-of-speech distributions for homographs (e.g., *light* as a noun, verb, or adjective).
        
        #### 2. Co-occurrence & Context
        * Lists top collocates grouped by position (immediately left or right) and displays sample concordance contexts.
    """,
    "Keyword": """
        ### 📖 Keyword Analysis Guide
        
        Keywords are words whose frequency in a target (study) corpus is statistically higher (or lower) than in a reference corpus.
        
        #### 1. Reference Corpus Selection
        * **Pre-built or Uploaded**: Select a reference corpus (e.g. BNC or Brown) or upload a custom text/frequency file to serve as the baseline comparison.
        
        #### 2. Settings & Analysis Basis
        * **P-Value Cutoff**: Restrict keywords to those meeting significance thresholds (e.g., 0.01 or 0.05).
        * **Analysis Basis (Optional)**:
          * **By Individual File**: Checked to generate and compare separate keyword lists for every unique file in the study corpus (e.g., creating 4 lists for 4 files).
          * **By Sub-corpora Attributes**: Checked to generate separate keyword lists for each XML attribute group (e.g. producing lists for every year, genre, or sentiment attribute value).
          * *Note: If these options are unchecked, the system calculates and displays only the 1 overall corpus-level keyword list, optimizing performance.*
        
        #### 3. Interpretation
        * **Positive Keywords (High Keyness)**: Words used significantly **more** in the target than the reference (reflecting target-specific themes).
        * **Negative Keywords (Low Keyness)**: Words used significantly **less** or completely absent in the target.
    """,
    "Distribution": """
        ### 📖 Distribution User Guide
        
        The Distribution module displays how words, lemmas, or metadata tags are distributed across different segments of your corpus:
        
        #### 1. Variable Mapping
        * Select an target query word and choose a metadata attribute (e.g., *genre*, *year*, *subcorpus*) as the mapping axis.
        * Displays absolute and relative frequencies partitioned by each sub-corpus category.
        
        #### 2. Plotting & Exporting
        * Renders bar charts and distribution tables to visualize frequency variations, and supports Excel export.
    """,
    "Statistical Testing": """
        ### 📖 Statistical Testing User Guide
        
        This module applies formal hypothesis testing to check if differences in word counts or readability metrics are statistically significant.
        
        #### 1. Selecting Test Variables
        * Choose the categories/sub-corpora to compare, and set the target frequencies or readability indices.
        * **Test Options**: Runs **Chi-Square Tests** for categorical distributions, or **t-tests / Mann-Whitney U** for numerical metrics.
        
        #### 2. Interpreting p-values
        * The interface highlights the test statistic, degrees of freedom, and the **p-value**. If $p < 0.05$, the differences are marked as statistically significant.
    """,
    "Summarisation": """
        ### 📖 Summarisation User Guide
        
        Generate concise summaries of documents or sections using natural language processing:
        
        #### 1. Input Source
        * Select specific documents, XML structural sections, or load custom text buffers.
        * **Length Adjuster**: Use sliders to define the target sentence limit or word count range for the summary output.
        
        #### 2. Model Selection
        * Choose between extractive summarizers (selecting key source sentences) or generative AI summarization.
    """,
    "Quiz Creation": """
        ### 📖 Quiz Creation User Guide
        
        Automatically compile interactive vocabulary and grammar quizzes based on your active corpus texts:
        
        #### 1. Question Types
        * Select question structures: **Multiple Choice**, **Fill in the Blanks**, or **Matching Definitions**.
        
        #### 2. Vocabulary & PASSAGE Selection
        * Define target words or choose source reading passages. The system automatically extracts distractors from the corpus vocabulary to build options.
        * **Export Quizzes**: Save questions and answer keys to text files or PDFs.
    """
}

def render_guidelines(module_name, sub_tab=None, key_prefix=""):
    """
    Renders guidelines in a side-by-side sticky column if guidelines are toggled on.
    Returns: col_main, col_guide
    """
    # Guidelines Toggle Button
    state_key = f'{module_name.lower().replace(" ", "_")}_show_guidelines'
    show_guide = get_state(state_key, False)
    
    if st.button("📖 Show Guidelines" if not show_guide else "✖ Hide Guidelines", key=f"btn_guide_{module_name.lower().replace(' ', '_')}_{key_prefix}"):
        show_guide = not show_guide
        set_state(state_key, show_guide)
        st.rerun()

    if show_guide:
        col_main, col_guide = st.columns([5, 3])
        with col_guide:
            st.markdown("""
            <style>
            div[data-testid="column"]:has(.sticky-guidelines) {
                position: -webkit-sticky !important;
                position: sticky !important;
                top: 80px !important;
                align-self: flex-start !important;
            }
            </style>
            <div class="sticky-guidelines"></div>
            """, unsafe_allow_html=True)
            with st.container(border=True):
                # Retrieve the markdown content
                if module_name == "Overview" and sub_tab:
                    content = GUIDELINES.get("Overview", {}).get(sub_tab, "No instructions available.")
                else:
                    content = GUIDELINES.get(module_name, "No instructions available.")
                st.markdown(content)
    else:
        col_main = st.container()
        
    return col_main
