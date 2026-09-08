import duckdb
import os
import html

def export_db_to_vertical_xml(db_path):
    """
    Reads a CORTEX DuckDB corpus file and exports it as a TreeTagger-style Vertical XML format.
    Dynamically discovers and includes all pre-existing and CORTEX-generated annotations 
    (POS, Lemma, Sentiment, Topic, NER, Dependencies, Readability, XML attributes, etc.).
    """
    try:
        con = duckdb.connect(db_path, read_only=True)
        # Get column names
        cols_info = con.execute("DESCRIBE corpus").fetchall()
        cols = [c[0] for c in cols_info]
        cols_lower = [c.lower() for c in cols]
        
        # 1. Standard token columns (always output first: token, pos, lemma)
        output_token_cols = []
        for col_name in ['token', 'pos', 'lemma']:
            if col_name in cols:
                output_token_cols.append(col_name)
                
        # 2. Known token-level annotation columns
        known_extra_token_cols = ['dep_rel', 'dep_head_id', 'dep_head_token', 'ent_type']
        for col_name in known_extra_token_cols:
            if col_name in cols and col_name not in output_token_cols:
                output_token_cols.append(col_name)
                
        # 3. Discover XML / segment attributes from database
        from core.preprocessing.xml_parser import get_xml_attribute_columns
        xml_attrs = get_xml_attribute_columns(con)
        
        known_sent_attrs = [
            'sentiment', 'topic', 'reading_ease_level', 'cefr_level',
            'genre', 'domain', 'author', 'title', 'date', 'source', 'year', 'doc_id', 'text_id'
        ]
        
        sent_attrs = []
        for attr in known_sent_attrs + xml_attrs:
            if attr in cols and attr not in sent_attrs:
                sent_attrs.append(attr)

        # 4. Discover any remaining custom columns in corpus table that are token-level annotations
        internal_cols = {
            'id', '_token_low', 'filename', 'sent_id', 'token', 'pos', 'lemma',
            'in_ner_start', 'ner_len', 'dep_rel', 'dep_head_id', 'dep_head_token', 'ent_type',
            '_conllu_id'
        }
        for c in cols:
            c_low = c.lower()
            if (c not in output_token_cols 
                and c not in sent_attrs 
                and c not in internal_cols
                and not c_low.startswith('in_')
                and not c_low.endswith('_len')
                and not c_low.endswith('_start')
                and not c_low.endswith('_id')):
                
                # Check if it's ner_<category>
                if c_low.startswith('ner_'):
                    continue
                output_token_cols.append(c)

        # NER tracking structure (from ner_service)
        has_ner = 'in_ner_start' in cols and 'ner_len' in cols
        
        # Discover generic inline tags (e.g. in_<tag>_start and <tag>_len)
        inline_tag_cols = []
        for c in cols:
            c_low = c.lower()
            if c_low.startswith('in_') and c_low.endswith('_start') and c_low != 'in_ner_start':
                tag_base = c_low[3:-6]
                len_col = f"{tag_base}_len"
                if len_col in cols_lower:
                    # Find actual column name for length
                    actual_len_col = cols[cols_lower.index(len_col)]
                    inline_tag_cols.append((c, actual_len_col, tag_base.upper()))
        
        # Get filenames
        filenames = con.execute("SELECT DISTINCT filename FROM corpus WHERE filename IS NOT NULL").fetchall()
        filenames = [f[0] for f in filenames if f[0]]
        if not filenames:
            filenames = ['corpus']
            
        lines = []
        
        for fname in filenames:
            lines.append(f'<text filename="{html.escape(fname)}">')
            
            # Get sentences for this file ordered by appearance
            if fname == 'corpus':
                query = "SELECT sent_id FROM corpus GROUP BY sent_id ORDER BY MIN(id)"
                sents = con.execute(query).fetchall()
            else:
                query = "SELECT sent_id FROM corpus WHERE filename = ? GROUP BY sent_id ORDER BY MIN(id)"
                sents = con.execute(query, [fname]).fetchall()
                
            for s in sents:
                sent_id = s[0]
                
                # Get all tokens for this sentence
                if fname == 'corpus':
                    q = "SELECT * FROM corpus WHERE sent_id = ? ORDER BY id"
                    rows = con.execute(q, [sent_id]).fetchall()
                else:
                    q = "SELECT * FROM corpus WHERE filename = ? AND sent_id = ? ORDER BY id"
                    rows = con.execute(q, [fname, sent_id]).fetchall()
                    
                if not rows: continue
                
                # Extract sentence level attributes from the first row
                attr_str = ""
                for attr in sent_attrs:
                    idx = cols.index(attr)
                    val = rows[0][idx]
                    if val is not None and str(val).strip() != "":
                        clean_val = html.escape(str(val))
                        attr_str += f' {attr}="{clean_val}"'
                        
                lines.append(f'<s id="{sent_id}"{attr_str}>')
                
                ner_active_tag = None
                ner_countdown = 0
                
                # Track generic inline tags
                active_inline_tags = {} # tag_base -> countdown
                
                for row in rows:
                    # Handle generic inline tags (open tag)
                    for start_col, len_col, tag_name in inline_tag_cols:
                        st_val = row[cols.index(start_col)]
                        if st_val is not None and str(st_val).strip() != "":
                            len_val = row[cols.index(len_col)]
                            try: cnt = int(len_val)
                            except: cnt = 1
                            active_inline_tags[tag_name] = cnt
                            lines.append(f"<{tag_name}>")

                    # Handle NER tags (open tag)
                    if has_ner:
                        ner_start = row[cols.index('in_ner_start')]
                        ner_len = row[cols.index('ner_len')]
                        if ner_start is not None and str(ner_start).strip() != "":
                            tag_label = None
                            if 'ent_type' in cols:
                                ent_val = row[cols.index('ent_type')]
                                if ent_val and str(ent_val).strip() and str(ent_val).upper() != 'TRUE':
                                    tag_label = str(ent_val).strip()
                            if not tag_label:
                                for c_name in cols:
                                    if c_name.lower().startswith('ner_') and c_name.lower() not in ('ner_len', 'ner_id'):
                                        val = row[cols.index(c_name)]
                                        if val and str(val).strip():
                                            tag_label = c_name[4:].upper()
                                            break
                            if not tag_label or tag_label.upper() == 'TRUE':
                                tag_label = str(ner_start).strip()
                            if not tag_label or tag_label.upper() == 'TRUE':
                                tag_label = "NER"
                                
                            ner_active_tag = tag_label.replace(" ", "_")
                            try:
                                ner_countdown = int(ner_len)
                            except:
                                ner_countdown = 1
                            lines.append(f"<{ner_active_tag}>")
                    
                    # Token output (token \t pos \t lemma \t extra1 ...)
                    token_vals = []
                    for c in output_token_cols:
                        val = row[cols.index(c)]
                        token_vals.append(str(val) if val is not None else "_")
                    
                    lines.append("\t".join(token_vals))
                    
                    # Handle NER tags (close tag)
                    if ner_countdown > 0:
                        ner_countdown -= 1
                        if ner_countdown == 0 and ner_active_tag:
                            lines.append(f"</{ner_active_tag}>")
                            ner_active_tag = None
                            
                    # Handle generic inline tags (close tag)
                    closed_tags = []
                    for tag_name, cnt in list(active_inline_tags.items()):
                        cnt -= 1
                        if cnt == 0:
                            lines.append(f"</{tag_name}>")
                            closed_tags.append(tag_name)
                        else:
                            active_inline_tags[tag_name] = cnt
                    for t in closed_tags:
                        del active_inline_tags[t]
                            
                lines.append('</s>')
                
            lines.append('</text>')
            
        con.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating XML: {e}"
