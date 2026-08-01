import duckdb
import os

def export_db_to_vertical_xml(db_path):
    """
    Reads a CORTEX DuckDB corpus file and exports it as a TreeTagger-style Vertical XML format.
    Dynamically discovers and includes advanced annotations (sentiment, NER, readability, etc).
    """
    try:
        con = duckdb.connect(db_path)
        # Get column names
        cols_info = con.execute("DESCRIBE corpus").fetchall()
        cols = [c[0] for c in cols_info]
        
        # Known token level output columns
        output_token_cols = ['token', 'pos', 'lemma']
        for extra in ['dep_rel', 'dep_head_id', 'dep_head_token', 'ent_type']:
            if extra in cols:
                output_token_cols.append(extra)
                
        # Known sentence level attributes
        sent_attrs = []
        for extra in ['sentiment', 'topic', 'reading_ease_level']: 
            if extra in cols:
                sent_attrs.append(extra)
                
        # NER tracking structure (from ner_service)
        has_ner = 'in_ner_start' in cols and 'ner_len' in cols
        
        # Get filenames
        filenames = con.execute("SELECT DISTINCT filename FROM corpus WHERE filename IS NOT NULL").fetchall()
        filenames = [f[0] for f in filenames if f[0]]
        if not filenames:
            filenames = ['corpus']
            
        lines = []
        
        for fname in filenames:
            lines.append(f'<text filename="{fname}">')
            
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
                    q = f"SELECT * FROM corpus WHERE sent_id = ? ORDER BY id"
                    rows = con.execute(q, [sent_id]).fetchall()
                else:
                    q = f"SELECT * FROM corpus WHERE filename = ? AND sent_id = ? ORDER BY id"
                    rows = con.execute(q, [fname, sent_id]).fetchall()
                    
                if not rows: continue
                
                # Extract sentence level attributes from the first row
                attr_str = ""
                for attr in sent_attrs:
                    idx = cols.index(attr)
                    val = rows[0][idx]
                    if val is not None and str(val).strip() != "":
                        attr_str += f' {attr}="{val}"'
                        
                lines.append(f'<s id="{sent_id}"{attr_str}>')
                
                ner_active_tag = None
                ner_countdown = 0
                
                for row in rows:
                    # Handle NER tags (open tag)
                    if has_ner:
                        ner_start = row[cols.index('in_ner_start')]
                        ner_len = row[cols.index('ner_len')]
                        if ner_start is not None and str(ner_start).strip() != "":
                            ner_active_tag = str(ner_start).replace(" ", "_")
                            try:
                                ner_countdown = int(ner_len)
                            except:
                                ner_countdown = 1
                            lines.append(f"<{ner_active_tag}>")
                    
                    # Token output
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
                            
                lines.append(f'</s>')
                
            lines.append(f'</text>')
            
        con.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating XML: {e}"
