import os
import codecs
import re
from core.preprocessing.acoustic_extractor import AcousticExtractor

def parse_textgrid_content(filepath):
    """
    Parses a TextGrid file (utf-8 or utf-16) and returns a dict of tiers and point tiers.
    """
    try:
        with codecs.open(filepath, 'r', 'utf-8', 'ignore') as f:
            content = f.read()
            if 'ooTextFile' not in content:
                raise UnicodeDecodeError("not utf-8", b"", 0, 1, "not utf-8")
    except UnicodeDecodeError:
        with codecs.open(filepath, 'r', 'utf-16', 'ignore') as f:
            content = f.read()

    interval_tiers = {}
    point_tiers = {}
    
    items = re.split(r'item \[\d+\]:', content)[1:]
    for item in items:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', item)
        class_match = re.search(r'class\s*=\s*"([^"]*)"', item)
        if not name_match or not class_match: 
            continue
            
        tname = name_match.group(1)
        tclass = class_match.group(1)
        
        if tclass == 'IntervalTier':
            intervals = []
            for p in item.split('intervals [')[1:]:
                try:
                    xmin = float(re.search(r'xmin\s*=\s*([\d\.]+)', p).group(1))
                    xmax = float(re.search(r'xmax\s*=\s*([\d\.]+)', p).group(1))
                    text_match = re.search(r'text\s*=\s*"([^"]*)"', p)
                    text = text_match.group(1) if text_match else ""
                    intervals.append({'start': xmin, 'end': xmax, 'text': text})
                except AttributeError:
                    continue
            interval_tiers[tname] = intervals
            
        elif tclass == 'TextTier':
            points = []
            for p in item.split('points [')[1:]:
                try:
                    num = float(re.search(r'number\s*=\s*([\d\.]+)', p).group(1))
                    mark_match = re.search(r'mark\s*=\s*"([^"]*)"', p)
                    mark = mark_match.group(1) if mark_match else ""
                    points.append({'time': num, 'mark': mark})
                except AttributeError:
                    continue
            point_tiers[tname] = points
            
    return interval_tiers, point_tiers

def textgrid_to_dataframe(filepath, audio_path=None):
    """
    Converts a TextGrid (and optional companion .wav) to a flat token-level dictionary list.
    """
    interval_tiers, point_tiers = parse_textgrid_content(filepath)
    
    # 1. Identify Word Tier (Backbone)
    # Usually "TA - words" or just "Words" or similar. Fallback to first interval tier.
    word_tier_name = None
    for name in interval_tiers.keys():
        if 'word' in name.lower():
            word_tier_name = name
            break
    if not word_tier_name and interval_tiers:
        word_tier_name = list(interval_tiers.keys())[0]
        
    if not word_tier_name:
        return []
        
    words = [w for w in interval_tiers[word_tier_name] if w['text'].strip() != '']
    
    # 2. Identify Phones Tier
    phone_tier_name = None
    for name in interval_tiers.keys():
        if 'phone' in name.lower() or 'phoneme' in name.lower():
            phone_tier_name = name
            break
    phones = interval_tiers.get(phone_tier_name, []) if phone_tier_name else []
    
    # 3. Identify Syllables Tier
    syl_tier_name = None
    for name in point_tiers.keys():
        if 'syllable' in name.lower():
            syl_tier_name = name
            break
    syllables = point_tiers.get(syl_tier_name, []) if syl_tier_name else []
    
    # 4. Structural Tiers (Utterance, Speaker, Comments)
    structural_tiers = {k: v for k, v in interval_tiers.items() if k not in [word_tier_name, phone_tier_name]}
    
    # 5. Acoustic Extractor
    acoustic = AcousticExtractor(audio_path) if audio_path else None
    
    # 6. Align!
    tokens_data = []
    
    # Utterance tracking (group words into sentences if they share a structural tier block)
    # We'll just use global_sent_id. To group, we check the primary structural tier (e.g. TA or ToneUnit)
    primary_struct_tier = None
    if structural_tiers:
        # Pick the one with the fewest empty intervals as primary (longest blocks)
        primary_struct_tier = list(structural_tiers.keys())[0] 

    eps = 0.005
    sent_id_counter = 1
    
    for w in words:
        w_start, w_end = w['start'], w['end']
        
        row = {
            'token': w['text'],
            'start_time': round(w_start, 3),
            'end_time': round(w_end, 3),
            'duration': round(w_end - w_start, 3),
            'sent_id': sent_id_counter,
            'pos': '##TAG', # Placeholder so it works with Cortex standard queries
            'lemma': w['text'].lower()
        }
        
        # Phones Overlap
        if phones:
            w_phones = [p['text'] for p in phones if p['text'] != '' and (p['start'] >= w_start - eps and p['end'] <= w_end + eps)]
            row['phones'] = " ".join(w_phones)
            
        # Syllables Count
        if syllables:
            w_syls = [p for p in syllables if w_start - eps <= p['time'] <= w_end + eps]
            row['syllable_count'] = len(w_syls)
            
        # Structural Tiers Overlap
        for st_name, st_intervals in structural_tiers.items():
            # Find which interval this word belongs to
            st_match = [s['text'] for s in st_intervals if s['text'] != '' and (w_start >= s['start'] - eps and w_end <= s['end'] + eps)]
            if st_match:
                # Clean up column name for DuckDB (no spaces/dashes)
                clean_name = st_name.replace(' ', '_').replace('-', '').lower()
                row[f'tier_{clean_name}'] = st_match[0]
                
        # Acoustic Features
        if acoustic:
            ac_feats = acoustic.get_features_for_interval(w_start, w_end)
            row.update(ac_feats)
            
        tokens_data.append(row)
        
    return tokens_data
