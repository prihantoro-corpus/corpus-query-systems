import os
import re
import tempfile
import pandas as pd
from core.visualiser.styles import POS_COLOR_MAP

def prepare_standalone_pyvis_html(net, height_px=550, bg_color="#222222"):
    """
    Generates completely self-contained, offline-compatible, dark-mode Pyvis HTML
    that will NEVER display a white box or fail due to external CDN or relative path issues.
    """
    raw_html = net.generate_html()

    # Remove broken relative script and stylesheet references
    raw_html = re.sub(r'<script[^>]*src=["\']?lib/bindings/utils\.js["\']?[^>]*>\s*</script>', '', raw_html)
    raw_html = re.sub(r'<script[^>]*src=["\']?\.\./node_modules/[^>]*>\s*</script>', '', raw_html)
    raw_html = re.sub(r'<link[^>]*href=["\']?\.\./node_modules/[^>]*>', '', raw_html)
    raw_html = re.sub(r'<link[^>]*bootstrap[^>]*>', '', raw_html)
    raw_html = re.sub(r'<script[^>]*bootstrap[^>]*></script>', '', raw_html)

    # Replace Bootstrap card wrapper
    raw_html = raw_html.replace(
        '<div class="card" style="width: 100%">',
        f'<div style="width: 100%; height: {height_px}px; background-color: {bg_color};">'
    )

    # Enforce dark mode CSS reset
    dark_css = f"""
    <style>
    *, *::before, *::after {{
        box-sizing: border-box !important;
    }}
    html, body {{
        background-color: {bg_color} !important;
        color: #ffffff !important;
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
        height: 100% !important;
        overflow: hidden !important;
    }}
    .card, .card-body {{
        background-color: {bg_color} !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }}
    #mynetwork {{
        width: 100% !important;
        height: {height_px}px !important;
        background-color: {bg_color} !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    </style>
    """
    if "</head>" in raw_html:
        raw_html = raw_html.replace("</head>", dark_css + "</head>")
    else:
        raw_html = dark_css + raw_html

    return raw_html

def create_pyvis_graph(target_word, coll_df, measure_col="LL", measure_name="LL", font_size=26):
    try:
        from pyvis.network import Network
    except ImportError:
        return ""

    net = Network(height="550px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='in_line')
    if coll_df.empty: return ""
    max_score = coll_df[measure_col].max()
    min_score = coll_df[measure_col].min()
    score_range = max_score - min_score
    
    target_font_size = int(font_size * 1.25)
    coll_font_size = int(font_size)

    net.set_options(f"""
    var options = {{
      "nodes": {{"borderWidth": 2, "size": 25, "font": {{"size": {coll_font_size}}}}},
      "edges": {{"width": 4, "smooth": {{"type": "dynamic"}}}},
      "interaction": {{
        "zoomView": false,
        "navigationButtons": true,
        "hover": true,
        "dragNodes": true,
        "dragView": true
      }},
      "physics": {{
        "solver": "barnesHut",
        "barnesHut": {{
          "gravitationalConstant": -3500,
          "centralGravity": 0.3,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.9,
          "avoidOverlap": 0.3
        }},
        "stabilization": {{
          "enabled": true,
          "iterations": 100,
          "updateInterval": 25
        }}
      }}
    }}
    """)
    
    net.add_node(target_word, label=target_word, size=45, color='#FFFF00', title=f"Target: {target_word}", x=0, y=0, fixed=True, font={'size': target_font_size, 'color': 'black'})
    
    LEFT_BIAS = -500; RIGHT_BIAS = 500
    all_directions = coll_df['Direction'].unique()
    if 'R' not in all_directions and 'L' in all_directions: RIGHT_BIAS = -500
    elif 'L' not in all_directions and 'R' in all_directions: LEFT_BIAS = 500

    for index, row in coll_df.iterrows():
        collocate = row['Collocate']
        score_val = row[measure_col]
        observed = row['Observed']
        pos_tag = row['POS']
        direction = row.get('Direction', 'R') 
        obs_l = row.get('Obs_L', 0)
        obs_r = row.get('Obs_R', 0)
        x_position = LEFT_BIAS if direction in ('L', 'B') else RIGHT_BIAS

        pos_code = pos_tag[0].upper() if pos_tag and len(pos_tag) > 0 else 'O'
        if pos_tag.startswith('##'): pos_code = '#'
        elif pos_code not in ['N', 'V', 'J', 'R']: pos_code = 'O'
        
        color = POS_COLOR_MAP.get(pos_code, POS_COLOR_MAP['O'])
        
        node_size = 25
        if score_range > 0:
            normalized_score = (score_val - min_score) / score_range
            node_size = 15 + normalized_score * 25 
            
        tooltip_title = (
            f"POS: {row['POS']}\n"
            f"Obs: {observed} (Left: {obs_l}, Right: {obs_r})\n"
            f"{measure_name}: {score_val:.2f}\n"
            f"Dominant Direction: {direction}"
        )

        net.add_node(collocate, label=collocate, size=node_size, color=color, title=tooltip_title, x=x_position, font={'size': coll_font_size, 'color': 'white'})
        net.add_edge(target_word, collocate, value=score_val, width=5, title=f"{measure_name}: {score_val:.2f}")

    return prepare_standalone_pyvis_html(net, height_px=550, bg_color="#222222")
