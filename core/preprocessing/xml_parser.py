import xml.etree.ElementTree as ET
import re
import pandas as pd
try:
    from lxml import etree as LXML_ET
    HAS_LXML = True
except ImportError:
    HAS_LXML = False

from core.preprocessing.cleaning import sanitize_xml_content

def extract_xml_structure(xml_input, max_values=20):
    """
    Parses XML content and extracts structure.
    Returns (structure, error_message)
    """
    if xml_input is None:
        return None, None
        
    cleaned_xml_content = None
    if isinstance(xml_input, str):
        cleaned_xml_content = xml_input
    else:
        try:
            xml_input.seek(0)
            xml_content = xml_input.read().decode('utf-8', errors='ignore')
            cleaned_xml_content = sanitize_xml_content(xml_content)
        except Exception as e:
            return None, f"File Read Error: {e}"

    if not cleaned_xml_content:
        return None, "Empty content"

    try:
        if HAS_LXML:
            parser = LXML_ET.XMLParser(recover=True, encoding='utf-8')
            root = LXML_ET.fromstring(cleaned_xml_content.encode('utf-8'), parser=parser)
        else:
            root = ET.fromstring(cleaned_xml_content) 
    except Exception as e:
        return None, f"XML Parsing Error: {e}"

    structure = {}
    
    def process_element(element):
        tag = element.tag
        if tag not in structure:
            structure[tag] = {}
        
        for attr_name, attr_value in element.attrib.items():
            if attr_name not in structure[tag]:
                structure[tag][attr_name] = set()
            
            if len(structure[tag][attr_name]) < max_values:
                structure[tag][attr_name].add(attr_value)

        for child in element:
            process_element(child)

    process_element(root)
    return structure, None

def parse_xml_with_inline_tags(element, context_tags, tokens_data, state, combined_attrs, tag_counters, stanza_processor=None, lang_code='en', is_uam_xml=False, uam_active_phrase_tags=None, uam_id_map=None):
    """
    Recursively parse XML element, preserving inline tag context.
    Supports UAM Corpus Tool XML mode with stand-off phrase tag inheritance.
    
    Args:
        element: XML element to parse
        context_tags: Dict of current tag context
        tokens_data: List to append token records to
        state: State dictionary tracking sentence ID
        combined_attrs: Segment-level attributes
        tag_counters: Global dict tracking instance IDs
        stanza_processor: Optional Stanza tagging function
        lang_code: Language code
        is_uam_xml: Whether to apply UAM Corpus Tool stand-off phrase propagation
        uam_active_phrase_tags: Active phrase-level tags inherited across sibling tokens
        uam_id_map: Pre-indexed UAM segment attributes by ID
    """
    current_context = context_tags.copy()
    current_uam_phrase = (uam_active_phrase_tags or {}).copy()
    tag_name = element.tag.lower()
    
    # Skip structural tags that shouldn't be tracked as inline context
    structural_tags = {'corpus', 'text', 's', 'sent', 'u', 'utterance', 'p', 'para', 'ab', 'div', 'w', 'document', 'body', 'header', 'article', 'essay'}
    
    # For structural tags WITH attributes (e.g. <s id="1" title="...">),
    # propagate their attributes as segment-level metadata into combined_attrs
    # so they appear as columns in DuckDB and enable Restricted Search.
    excluded_meta_attrs = {'n', 'num', 'lang'}
    if tag_name in structural_tags and element.attrib:
        for k, v in element.attrib.items():
            clean_k = k.lower()
            if clean_k in excluded_meta_attrs:
                continue
            col_name = f"{tag_name}_id" if clean_k == 'id' else clean_k
            combined_attrs[col_name] = str(v).strip()

    if tag_name not in structural_tags:
        # Add boolean flag for tag presence (normalize tag name)
        current_context[f"in_{tag_name}"] = True
        
        # Track unique instance ID for this tag
        tag_counters[tag_name] = tag_counters.get(tag_name, 0) + 1
        current_context[f"{tag_name}_id"] = tag_counters[tag_name]
        
        # Add attributes prefixed by tag name
        attribs = dict(element.attrib)
        if tag_name == 'segment' and 'features' in attribs:
            feat_raw = str(attribs['features'])
            parts = feat_raw.split(';')
            attribs['domain'] = parts[0].replace('-', '_') if len(parts) >= 1 and parts[0].strip() else "none"
            attribs['category'] = parts[1].replace('-', '_') if len(parts) >= 2 and parts[1].strip() else "none"
            attribs['subcategory'] = parts[2].replace('-', '_') if len(parts) >= 3 and parts[2].strip() else "none"
            attribs['tag'] = parts[3].replace('-', '_') if len(parts) >= 4 and parts[3].strip() else "none"
            attribs['features'] = feat_raw.replace(';', '_').replace('-', '_')

            # In UAM XML mode, resolve parent segment attributes if referenced via parent='ID'
            if is_uam_xml:
                parent_id = attribs.get('parent')
                if parent_id and uam_id_map and parent_id in uam_id_map:
                    p_attribs = uam_id_map[parent_id]
                    if 'features' in p_attribs:
                        p_feat_raw = str(p_attribs['features'])
                        p_parts = p_feat_raw.split(';')
                        p_cat = p_parts[1].replace('-', '_') if len(p_parts) >= 2 and p_parts[1].strip() else "none"
                        p_tag = p_parts[3].replace('-', '_') if len(p_parts) >= 4 and p_parts[3].strip() else "none"
                        
                        attribs['parent_tag'] = p_tag
                        attribs['parent_category'] = p_cat
                        current_context['segment_tag'] = p_tag
                        current_context['segment_category'] = p_cat
                        current_uam_phrase['segment_tag'] = p_tag
                        current_uam_phrase['segment_category'] = p_cat

                cat = attribs['category'].lower()
                tag = attribs['tag'].lower()
                if 'frasa' in cat or tag in ['nom0', 'vrbx', 'prep0', 'yg0', 'nom', 'vrb', 'prep']:
                    for k, v in attribs.items():
                        clean_v = str(v).replace(';', '_').strip()
                        current_uam_phrase[f"{tag_name}_{k.lower()}"] = clean_v
                    current_uam_phrase[f"in_{tag_name}"] = True

        for k, v in attribs.items():
            clean_v = str(v).replace(';', '_').strip()
            current_context[f"{tag_name}_{k.lower()}"] = clean_v

    # Merge active UAM phrase tags into context if UAM XML mode is enabled
    if is_uam_xml and current_uam_phrase:
        for k, v in current_uam_phrase.items():
            if k not in current_context:
                current_context[k] = v
    

    # Helper to tokenize and add text with current context
    def tokenize_and_add(text, context):
        if not text or not text.strip():
            return
            
        # 1. Detect if this block of text is in Vertical Format (TreeTagger: token\ttag\tlemma)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        is_vertical = False
        if len(lines) > 0:
            # Strictly require tabs for auto-detection of vertical format inside inline tags
            # to avoid false positives on horizontal segments with few words.
            # Lowered threshold to 0.4 to be more inclusive of partially sparse vertical tags
            vertical_score = sum(1 for l in lines if '\t' in l) / len(lines)
            if vertical_score > 0.4:
                is_vertical = True

        if is_vertical:
            # Only use Pandas read_csv if the block of text is very large (e.g. > 1000 lines)
            # to avoid the high overhead of StringIO/DataFrame construction on small sentences.
            if len(lines) > 1000:
                from io import StringIO
                import pandas as pd
                
                # Fast-path: Vectorized parsing with Pandas
                try:
                    df_block = pd.read_csv(
                        StringIO(text), 
                        sep='\t', 
                        header=None, 
                        names=['token', 'pos', 'lemma'],
                        engine='c', 
                        dtype=str, 
                        quoting=3 # QUOTE_NONE
                    )
                    
                    if not df_block.empty:
                        # Broadcast segment-level metadata and sentence ID
                        df_block['sent_id'] = sent_id
                        for k, v in combined_attrs.items():
                            df_block[k] = v
                        for k, v in context.items():
                            df_block[k] = v
                        
                        tokens_data.extend(df_block.to_dict('records'))
                    return
                except Exception:
                    # Fallback to manual loop if Pandas fails
                    pass

            # Pre-merge attributes for massive performance gain in loop
            merged_meta = combined_attrs.copy()
            merged_meta.update(context)
            meta_keys = list(merged_meta.keys())
            meta_vals = [merged_meta[k] for k in meta_keys]
            
            has_lines = False
            for line in lines:
                parts = line.strip().split('\t')
                if not parts or not parts[0]: continue
                has_lines = True
                token = parts[0]
                pos = parts[1] if len(parts) > 1 else "TAG"
                lemma = parts[2] if len(parts) > 2 else parts[0]
                d = {'token': token, 'pos': pos, 'lemma': lemma, 'sent_id': state['global_sent_id']}
                for i in range(len(meta_keys)):
                    d[meta_keys[i]] = meta_vals[i]
                tokens_data.append(d)
            if has_lines: state['global_sent_id'] += 1
            return

        # 2. Use Stanza if available (for horizontal/inline text)
        if stanza_processor:
            try:
                tagged_data, err = stanza_processor(text, lang_code)
                if not err and tagged_data:
                    for rec in tagged_data:
                        local_sent = rec.get('sent_id', 1)
                        if 'current_local_sent' not in state: state['current_local_sent'] = local_sent
                        if local_sent != state['current_local_sent']:
                            state['global_sent_id'] += 1
                            state['current_local_sent'] = local_sent

                        row = {
                            'token': rec['token'],
                            'pos': rec['pos'],
                            'lemma': rec['lemma'],
                            'sent_id': state['global_sent_id']
                        }
                        row.update(combined_attrs)
                        row.update(context)
                        tokens_data.append(row)
                        
                    state['global_sent_id'] += 1
                    if 'current_local_sent' in state: del state['current_local_sent']
                    return
            except Exception:
                pass
        
        # 3. Fallback: simple whitespace tokenization
        cleaned_text = re.sub(r'([^\w\s])', r' \1 ', text)
        tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
        if tokens:
            for token in tokens:
                row = {
                    'token': token,
                    'pos': '##TAG',
                    'lemma': token,
                    'sent_id': state['global_sent_id']
                }
                row.update(combined_attrs)
                row.update(context)
                tokens_data.append(row)
            state['global_sent_id'] += 1
    
    # Process text BEFORE first child
    tokens_before = len(tokens_data)
    if element.text:
        tokenize_and_add(element.text, current_context)
    
    # Process children recursively
    for child in element:
        parse_xml_with_inline_tags(child, current_context, tokens_data, state, combined_attrs, tag_counters, stanza_processor, lang_code, is_uam_xml, current_uam_phrase, uam_id_map)
        
        # Process tail text (text AFTER child tag but inside parent)
        if child.tail:
            tokenize_and_add(child.tail, current_context)
    
    # Calculate length (number of tokens) for this tag instance
    tokens_after = len(tokens_data)
    tag_inner_len = tokens_after - tokens_before
    
    if tag_name not in structural_tags and tag_inner_len > 0:
        # Mark the FIRST token of this tag with the start flag and length
        first_token_row = tokens_data[tokens_before]
        first_token_row[f"in_{tag_name}_start"] = True
        first_token_row[f"{tag_name}_len"] = tag_inner_len

def parse_xml_content_to_df(xml_input, force_vertical_xml=False, stanza_processor=None, lang_code='en', preserve_inline_tags=True, is_uam_xml=False):
    """
    Parses XML content, extracts sentences and IDs, and tokenizes/verticalizes.
    Returns dict with keys: lang_code, df_data, sent_map, attributes, error
    """
    cleaned_xml_content = None
    if isinstance(xml_input, str):
        cleaned_xml_content = sanitize_xml_content(xml_input)
    else:
        try:
            xml_input.seek(0)
            xml_content = xml_input.read().decode('utf-8', errors='ignore')
            cleaned_xml_content = sanitize_xml_content(xml_content)
        except Exception as e:
            return {'error': f"Error reading XML file: {e}"}
    
    if not cleaned_xml_content:
        return {'error': "Empty content"}
    
    try:
        if HAS_LXML:
            parser = LXML_ET.XMLParser(recover=True, encoding='utf-8')
            root = LXML_ET.fromstring(cleaned_xml_content.encode('utf-8'), parser=parser)
        else:
            root = ET.fromstring(cleaned_xml_content)
            
        xml_lang = root.get('lang')
        if not xml_lang:
            lang_match = re.search(r'(<text\s+lang="([^"]+)">|<corpus\s+[^>]*lang="([^"]+)">)', cleaned_xml_content)
            if lang_match:
                xml_lang = lang_match.group(3) or lang_match.group(2)
        
        final_lang = xml_lang.lower() if xml_lang else lang_code.lower()
        if not final_lang: final_lang = 'en'
            
    except Exception as e:
        return {'error': f"Tokenization Parse Error: {e}"}

    df_data = []
    sent_map = {}
    detected_attrs = {} 
    sequential_id_counter = 0
    
    excluded_attrs = ('n', 'num', 'lang') # Removed 'id' from exclusion, manually handled below
    base_root_attrs = {}
    for k, v in root.attrib.items():
        clean_k = re.sub(r'\{.*?\}', '', k)
        if clean_k.lower() in excluded_attrs: continue
        key_name = 'doc_id' if clean_k.lower() == 'id' else clean_k
        base_root_attrs[key_name] = v
    
    for k, v in base_root_attrs.items():
        if k not in detected_attrs: detected_attrs[k] = set()
        detected_attrs[k].add(v)

    if preserve_inline_tags:
        tag_counters = {}
        state = {'global_sent_id': 1}
        uam_id_map = {}
        if is_uam_xml:
            for elem in root.findall('.//segment'):
                sid = elem.attrib.get('id')
                if sid:
                    uam_id_map[sid] = elem.attrib

        parse_xml_with_inline_tags(
            root, 
            {}, 
            df_data, 
            state, 
            base_root_attrs,
            tag_counters,
            stanza_processor,
            final_lang,
            is_uam_xml,
            None,
            uam_id_map
        )
        temp_sent_parts = {}
        for r in df_data:
            sid = r.get('sent_id')
            if sid not in sent_map:
                temp_sent_parts.setdefault(sid, []).append(r['token'])
        for sid, parts in temp_sent_parts.items():
            sent_map[sid] = " ".join(parts)
        return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}

    elements_to_process = []
    pass1_tags = {'sent', 's', 'u', 'utterance', 'turn'}
    pass2_tags = {'p', 'para', 'ab', 'div', 'dialogue'} 
    pass3_tags = {'text', 'body', 'document', 'article', 'essay', 'entry'}

    def traverse_and_collect(element, current_attrs, target_tags):
        new_attrs = current_attrs.copy()
        
        # Prepare attributes for this element, checking exclusions and renaming id
        elem_attrs = {}
        for k, v in element.attrib.items():
            clean_k = re.sub(r'\{.*?\}', '', k)
            if clean_k.lower() in excluded_attrs: continue
            key_name = 'doc_id' if clean_k.lower() == 'id' else clean_k
            elem_attrs[key_name] = v
            
        new_attrs.update(elem_attrs)
        
        if element.tag in target_tags:
            elements_to_process.append((element, new_attrs))
            return 
        for child in element:
            traverse_and_collect(child, new_attrs, target_tags)
            
    traverse_and_collect(root, {}, pass1_tags)
    if not elements_to_process:
        traverse_and_collect(root, {}, pass2_tags)
    if not elements_to_process:
         traverse_and_collect(root, {}, pass3_tags)

    if not elements_to_process:
        raw_sentence_text = "".join(root.itertext()).strip() 
        if raw_sentence_text:
            if stanza_processor:
                res = stanza_processor(raw_sentence_text, final_lang)
                if isinstance(res, tuple) and len(res) == 2:
                    stanza_records, err = res
                else:
                    stanza_records, err = res, None
                    
                if not err and stanza_records:
                    current_stanza_sent = -1
                    current_sent_text_parts = []
                    for rec in stanza_records:
                        if rec['sent_id'] != current_stanza_sent:
                            if current_stanza_sent != -1:
                                sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                            
                            sequential_id_counter += 1
                            current_stanza_sent = rec['sent_id']
                            current_sent_text_parts = []
                        
                        row = {"token": rec['token'], "pos": rec['pos'], "lemma": rec['lemma'], "sent_id": sequential_id_counter, "ent_type": rec.get('ent_type', '')}
                        row.update(base_root_attrs)
                        df_data.append(row)
                        current_sent_text_parts.append(rec['token'])
                    
                    if current_sent_text_parts:
                        sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                        
                    return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}

            cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_sentence_text) 
            tokens = [t.strip() for t in cleaned_text.split() if t.strip()]
            if tokens:
               for token in tokens:
                    row = {"token": token, "pos": "TAG", "lemma": token, "sent_id": 1, "ent_type": ""}
                    row.update(base_root_attrs)
                    df_data.append(row)
               sent_map[1] = raw_sentence_text
            return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}
        return {'error': "No parseable content found"}

    sequential_id_counter = 0
    tag_counters = {} # NEW: Track unique instance IDs across all elements

    for sent_elem, combined_row_attrs in elements_to_process:
        for k, v in combined_row_attrs.items():
            if k not in detected_attrs: detected_attrs[k] = set()
            detected_attrs[k].add(v)

        sent_id_str = sent_elem.get('n') or sent_elem.get('id')
        sent_id = None
        if sent_id_str:
            try: sent_id = int(sent_id_str)
            except ValueError:
                sequential_id_counter += 1
                sent_id = sequential_id_counter
        else:
            sequential_id_counter += 1
            sent_id = sequential_id_counter

        word_tags = sent_elem.findall('.//w')
        raw_sentence_text = ""
        
        # NEW: Use inline tag parser if enabled
        if preserve_inline_tags and not word_tags:
            parse_xml_with_inline_tags(
                sent_elem, 
                {}, # Start with empty tag context
                df_data, 
                {'global_sent_id': sent_id}, 
                combined_row_attrs,
                tag_counters,
                stanza_processor,
                final_lang
            )
            # Build sentence text for sent_map across all sent_ids generated
            temp_sent_parts = {}
            for r in df_data:
                sid = r.get('sent_id')
                if sid not in sent_map:
                    temp_sent_parts.setdefault(sid, []).append(r['token'])
            for sid, parts in temp_sent_parts.items():
                sent_map[sid] = " ".join(parts)
            continue  # Skip legacy processing
        
        # LEGACY: Original processing for <w> tags and vertical format
        if word_tags:
            raw_tokens = []
            for w_elem in word_tags:
                token = w_elem.text.strip() if w_elem.text else ""
                if not token: continue
                pos = w_elem.get('pos') or w_elem.get('type') or "TAG"
                lemma = w_elem.get('lemma') or token
                row = {"token": token, "pos": pos, "lemma": lemma, "sent_id": sent_id, "ent_type": ""}
                if combined_row_attrs: row.update(combined_row_attrs)
                df_data.append(row)
                raw_tokens.append(token)
            raw_sentence_text = " ".join(raw_tokens)
        else:
            raw_sentence_text = "".join(sent_elem.itertext()).strip() 
            inner_content = raw_sentence_text
            normalized_content = inner_content.replace('\r\n', '\n').replace('\r', '\n')
            lines = [line.strip() for line in normalized_content.split('\n') if line.strip()]
            
            is_vertical_format = False
            if lines:
                if force_vertical_xml: is_vertical_format = True
                else:
                    def is_line_vertical(l):
                        if '\t' in l: return True
                        words = re.split(r'\s+', l.strip())
                        return 1 <= len(words) <= 3 
                    is_vertical_format = sum(is_line_vertical(line) for line in lines) / len(lines) > 0.8
            
            if is_vertical_format:
                raw_tokens = []
                for line in lines:
                    parts = re.split(r'\t+', line.strip())
                    if not parts or not parts[0]: continue
                    token = parts[0]
                    pos = parts[1] if len(parts) > 1 else "TAG"
                    lemma = parts[2] if len(parts) > 2 else token
                    row = {"token": token, "pos": pos, "lemma": lemma, "sent_id": sent_id, "ent_type": ""}
                    if combined_row_attrs: row.update(combined_row_attrs)
                    df_data.append(row)
                    raw_tokens.append(token)
            else:
                raw_text_to_tokenize = raw_sentence_text.replace('\n', ' ').replace('\t', ' ')
                if stanza_processor:
                    res = stanza_processor(raw_text_to_tokenize, final_lang)
                    if isinstance(res, tuple) and len(res) == 2:
                        stanza_records, err = res
                    else:
                        stanza_records, err = res, None
                    
                    if not err and stanza_records:
                        current_stanza_sent = -1
                        current_sent_text_parts = []
                        for rec in stanza_records:
                            if rec['sent_id'] != current_stanza_sent:
                                if current_stanza_sent != -1:
                                    sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                                
                                sequential_id_counter += 1
                                current_stanza_sent = rec['sent_id']
                                current_sent_text_parts = []
                            
                            row = {"token": rec['token'], "pos": rec['pos'], "lemma": rec['lemma'], "sent_id": sequential_id_counter, "ent_type": rec.get('ent_type', '')}
                            if combined_row_attrs: row.update(combined_row_attrs)
                            df_data.append(row)
                            current_sent_text_parts.append(rec['token'])
                        
                        if current_sent_text_parts:
                            sent_map[sequential_id_counter] = " ".join(current_sent_text_parts)
                        continue
                
                cleaned_text = re.sub(r'([^\w\s])', r' \1 ', raw_text_to_tokenize) 
                tokens = [t.strip() for t in cleaned_text.split() if t.strip()] 
                for token in tokens:
                    row = {"token": token, "pos": "TAG", "lemma": token, "sent_id": sent_id, "ent_type": ""}
                    if combined_row_attrs: row.update(combined_row_attrs)
                    df_data.append(row)
        
        if raw_sentence_text:
            sent_map[sent_id] = raw_sentence_text.strip()
            
    if not df_data:
        return {'error': "No tokenized data extracted"}
        
    return {'lang_code': final_lang, 'df_data': df_data, 'sent_map': sent_map, 'attributes': detected_attrs}

def format_structure_data_hierarchical(structure_data, indent_level=0, max_values=20):
    """
    Formats the hierarchical XML structure data into an indented HTML list.
    """
    if not structure_data:
        return ""

    html_list = []
    
    def get_indent(level):
        return f'<span style="padding-left: {level * 1.5}em;">'

    for tag in sorted(structure_data.keys()):
        tag_data = structure_data[tag]
        tag_line = f'{get_indent(indent_level)}<span style="color: #6A5ACD; font-weight: bold;">&lt;{tag}&gt;</span></span><br>'
        html_list.append(tag_line)
        
        for attr in sorted(tag_data.keys()):
            values = sorted(list(tag_data.get(attr, set())))
            sampled_values_str = ", ".join(values[:max_values])
            if len(values) > max_values:
                sampled_values_str += f", ... ({len(values) - max_values} more unique)"

            attr_line = f'{get_indent(indent_level + 1)}'
            attr_line += f'<span style="color: #8B4513;">@{attr}</span> = '
            attr_line += f'<span style="color: #3CB371;">"{sampled_values_str}"</span></span><br>'
            html_list.append(attr_line)

    return "".join(html_list)

def get_xml_attribute_columns(con):
    """Identifies columns in the DuckDB corpus table that are XML segment-level attributes.
    Excludes internal tracking columns generated by the inline tag parser (in_*, *_len, *_start, *_id).
    """
    try:
        cols_info = con.execute("PRAGMA table_info(corpus)").fetchall()
        db_cols = [c[1] for c in cols_info]
        standard_cols = {
            'token', 'pos', 'lemma', 'sent_id', '_token_low', 'id', 'filename',
            'dep_rel', 'dep_head_id', 'dep_head_token', '_conllu_id'
        }
        internal_suffixes = ('_len', '_start', '_id')
        internal_prefixes = ('in_',)
        return [
            col for col in db_cols
            if col not in standard_cols
            and not col.endswith(internal_suffixes)
            and not any(col.startswith(pfx) for pfx in internal_prefixes)
        ]
    except:
        return []

def is_integer_col(con, col_name):
    """Checks if a column in the corpus table is purely integer-like (ignoring NULLs)."""
    try:
        sql = f'SELECT count(*) FROM corpus WHERE "{col_name}" IS NOT NULL AND TRY_CAST("{col_name}" AS BIGINT) IS NULL'
        fail_count = con.execute(sql).fetchone()[0]
        return fail_count == 0
    except:
        return False

def apply_xml_restrictions(filters):
    """
    Returns a SQL WHERE clause fragment based on user-selected XML attribute filters.
    Returns (sql_fragment, params_list)
    """
    if not filters:
        return "", []
    
    clauses = []
    params = []
    for attr, val_data in filters.items():
        if val_data['type'] == 'list':
            vals = val_data['values']
            placeholders = ', '.join(['?'] * len(vals))
            clauses.append(f'"{attr}" IN ({placeholders})')
            params.extend(vals)
        elif val_data['type'] == 'range':
            min_v = val_data['min']
            max_v = val_data['max']
            clauses.append(f'TRY_CAST("{attr}" AS BIGINT) BETWEEN ? AND ?')
            params.extend([min_v, max_v])
            
    return " AND " + " AND ".join(clauses), params


def parse_eaf_content_to_df_records(xml_content, stanza_processor=None, lang_code='en', filename='file.eaf', eaf_main_tier=None, eaf_gloss_tier=None, eaf_trans_tier=None):
    """
    Parses ELAN .eaf XML into aligned 7-layer token records for CORTEX.
    If explicit tier mappings are provided, uses them. Otherwise relies on heuristics.
    """
    if isinstance(xml_content, str):
        root = ET.fromstring(xml_content.encode('utf-8'))
    else:
        root = ET.fromstring(xml_content)
    
    tier_map = {}
    for tier in root.findall('TIER'):
        tier_id = tier.attrib.get('TIER_ID', '')
        tier_map[tier_id] = tier
        
    def get_ref_map(tier_name):
        res = {}
        if tier_name in tier_map:
            for ann in tier_map[tier_name].findall('ANNOTATION/REF_ANNOTATION'):
                ref = ann.attrib.get('ANNOTATION_REF', '')
                val = ann.find('ANNOTATION_VALUE').text or ''
                res[ref] = val
        return res

    # 1. Identify Root Orthographic Tier
    root_tier_id = eaf_main_tier
    if not root_tier_id:
        for t_id, tier in tier_map.items():
            if 'PARENT_REF' not in tier.attrib and tier.attrib.get('LINGUISTIC_TYPE_REF') in ['orthography', 'orthographic', 'transcription', 'ORT-F']:
                root_tier_id = t_id
                break
        if not root_tier_id:
            for t_id, tier in tier_map.items():
                if 'PARENT_REF' not in tier.attrib:
                    root_tier_id = t_id
                    break

    if not root_tier_id:
        return []

    # Root annotations
    root_annos = {}
    for ann in tier_map[root_tier_id].findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
        aid = ann.attrib.get('ANNOTATION_ID', '')
        val = ann.find('ANNOTATION_VALUE').text or ''
        root_annos[aid] = val

    # Parse ELAN <HEADER> <PROPERTY> metadata if present (e.g. sex, location, first_language, speaker)
    header_metadata = {}
    for prop in root.findall('HEADER/PROPERTY'):
        prop_name = prop.attrib.get('NAME', '').strip().lower().replace(' ', '_')
        prop_val = prop.text.strip() if prop.text else ''
        if prop_name and prop_val:
            header_metadata[prop_name] = prop_val

    # Flexible multilingual & custom tier lookup helper
    def find_tier_map(possible_names, ling_type_keywords=[]):
        # 1. Exact or case-insensitive match on TIER_ID
        for name in possible_names:
            for t_id in tier_map:
                if t_id.lower() == name.lower():
                    return get_ref_map(t_id)
        # 2. Substring match on TIER_ID
        for name in possible_names:
            for t_id in tier_map:
                if name.lower() in t_id.lower():
                    return get_ref_map(t_id)
        # 3. Match on LINGUISTIC_TYPE_REF attribute
        if ling_type_keywords:
            for t_id, tier in tier_map.items():
                ling_ref = tier.attrib.get('LINGUISTIC_TYPE_REF', '').lower()
                for kw in ling_type_keywords:
                    if kw.lower() in ling_ref:
                        return get_ref_map(t_id)
        return {}

    # Dependent tier maps with broad Indonesian, English & linguistic alias coverage
    ort_d_map = find_tier_map(['ORT-D', 'Orthographic_Delineated', 'ort_d', 'delineated', 'ortografi_delineasi'], ['delineat', 'morpheme', 'morfem'])
    phn_f_map = find_tier_map(['PHN-F', 'Phonetic', 'phn_f', 'fonetik', 'fonetis'], ['phonetic', 'fonetik'])
    phn_d_map = find_tier_map(['PHN-D', 'Phonetic_Delineated', 'phn_d', 'fonetik_delineasi'], ['phonetic_d', 'fonetik_d'])
    
    if eaf_trans_tier: trans_map = find_tier_map([eaf_trans_tier])
    else: trans_map = find_tier_map(['TRANS', 'Free_Translation', 'Translation', 'trans', 'terjemahan', 'terjemah', 'arti'], ['translation', 'terjemahan', 'free'])
    
    if eaf_gloss_tier: gloss_map = find_tier_map([eaf_gloss_tier])
    else: gloss_map = find_tier_map(['GLOSS', 'Morphemic_Gloss', 'Gloss', 'gloss', 'terjemahan-morfem', 'glosa', 'morfem', 'morpheme'], ['gloss', 'morfem', 'glosa'])

    # Speaker metadata tiers (e.g. sex, location, first_language, speaker)
    sex_map = find_tier_map(['SEX', 'sex', 'gender', 'jenis_kelamin'], ['sex', 'gender'])
    loc_map = find_tier_map(['LOCATION', 'location', 'city', 'lokasi', 'tempat'], ['location', 'lokasi'])
    l1_map = find_tier_map(['FIRST_LANGUAGE', 'first_language', 'l1', 'bahasa_ibu', 'native_language'], ['language', 'bahasa'])

    # Also capture ALL custom/unmapped tiers in the EAF file dynamically
    custom_tier_maps = {}
    known_matched_tiers = {'ort-d', 'phn-f', 'phn-d', 'trans', 'gloss', 'sex', 'location', 'first_language', root_tier_id.lower() if root_tier_id else ''}
    for t_id in tier_map:
        if t_id.lower() not in known_matched_tiers:
            c_map = get_ref_map(t_id)
            if c_map:
                # Sanitize column name for DB indexing (replace dashes/spaces with underscores)
                clean_col = re.sub(r'\W+', '_', t_id.strip()).lower()
                custom_tier_maps[clean_col] = c_map

    # Subdivided Word Tokens tier
    word_tokens = []
    word_tier_id = None
    for t_id in ['Word_Tokens', 'Words', 'Morphemes', 'Tokens', 'ORT-D', 'morfem', 'kata']:
        if t_id in tier_map:
            word_tier_id = t_id
            break
    if not word_tier_id:
        for t_id, tier in tier_map.items():
            if tier.attrib.get('PARENT_REF') == root_tier_id and tier.attrib.get('LINGUISTIC_TYPE_REF') in ['word_subdivision', 'morpheme', 'word', 'words', 'morfem', 'kata']:
                word_tier_id = t_id
                break

    if word_tier_id:
        for ann in tier_map[word_tier_id].findall('ANNOTATION/REF_ANNOTATION'):
            wid = ann.attrib.get('ANNOTATION_ID', '')
            parent_id = ann.attrib.get('ANNOTATION_REF', '')
            val = ann.find('ANNOTATION_VALUE').text or ''
            word_tokens.append({'wid': wid, 'parent_id': parent_id, 'word': val})
    else:
        w_counter = 1
        for parent_id, text in root_annos.items():
            words = text.split()
            for w in words:
                word_tokens.append({'wid': f'w{w_counter}', 'parent_id': parent_id, 'word': w})
                w_counter += 1

    word_index_by_parent = {}
    for wt in word_tokens:
        pid = wt['parent_id']
        idx = word_index_by_parent.get(pid, 0)
        wt['word_idx'] = idx
        word_index_by_parent[pid] = idx + 1

    records = []
    sent_id_map = {parent_id: idx+1 for idx, parent_id in enumerate(root_annos.keys())}
    
    for wt in word_tokens:
        wid = wt['wid']
        parent_id = wt['parent_id']
        w_ort_f = wt['word']
        word_idx = wt['word_idx']
        sent_id = sent_id_map.get(parent_id, 1)
        
        s_ort_d = ort_d_map.get(parent_id, '')
        s_phn_f = phn_f_map.get(parent_id, '')
        s_phn_d = phn_d_map.get(parent_id, '')
        s_trans = trans_map.get(parent_id, '')
        
        w_ort_d = w_ort_f
        if s_ort_d:
            for chunk in s_ort_d.split():
                if chunk.replace('-', '').lower() == w_ort_f.lower():
                    w_ort_d = chunk
                    break
        
        # Get word-level phonetic annotations if they exist, otherwise fallback to space-separated sentence string
        w_phn_f = phn_f_map.get(wid, '')
        if not w_phn_f and s_phn_f:
            chunks = s_phn_f.split()
            if word_idx < len(chunks):
                w_phn_f = chunks[word_idx]
                
        w_phn_d = phn_d_map.get(wid, '')
        if not w_phn_d and s_phn_d:
            chunks = s_phn_d.split()
            if word_idx < len(chunks):
                w_phn_d = chunks[word_idx]

        w_gloss = gloss_map.get(wid, '') or gloss_map.get(parent_id, '')
        
        w_sex = sex_map.get(parent_id, '') or sex_map.get(wid, '') or header_metadata.get('sex', '') or header_metadata.get('gender', '')
        w_loc = loc_map.get(parent_id, '') or loc_map.get(wid, '') or header_metadata.get('location', '') or header_metadata.get('city', '')
        w_l1 = l1_map.get(parent_id, '') or l1_map.get(wid, '') or header_metadata.get('first_language', '') or header_metadata.get('l1', '')

        rec = {
            'token': w_ort_f,
            'pos': 'TAG',
            'lemma': w_ort_f.lower(),
            'sex': w_sex,
            'location': w_loc,
            'first_language': w_l1,
            'sent_id': sent_id,
            'filename': filename
        }
        
        if w_ort_d: rec['ort_d'] = w_ort_d
        if w_phn_f: rec['phn_f'] = w_phn_f
        if w_phn_d: rec['phn_d'] = w_phn_d
        if w_gloss: rec['gloss'] = w_gloss
        if s_trans: rec['trans'] = s_trans
        
        # Attach any non-standard custom tiers dynamically to the DB record
        for c_col, c_map in custom_tier_maps.items():
            rec[c_col] = c_map.get(wid, '') or c_map.get(parent_id, '')

        records.append(rec)
        
    return records


