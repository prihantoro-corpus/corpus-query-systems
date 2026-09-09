import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import re
try:
    from core.visualiser.network import prepare_standalone_pyvis_html
except ImportError:
    def prepare_standalone_pyvis_html(net, height_px=550, bg_color="#0f172a"):
        raw_html = net.generate_html()
        raw_html = re.sub(r'<script[^>]*src=["\']?lib/bindings/utils\.js["\']?[^>]*>\s*</script>', '', raw_html)
        raw_html = re.sub(r'<script[^>]*src=["\']?\.\./node_modules/[^>]*>\s*</script>', '', raw_html)
        raw_html = re.sub(r'<link[^>]*href=["\']?\.\./node_modules/[^>]*>', '', raw_html)
        raw_html = re.sub(r'<link[^>]*bootstrap[^>]*>', '', raw_html)
        raw_html = re.sub(r'<script[^>]*bootstrap[^>]*></script>', '', raw_html)
        raw_html = raw_html.replace('<div class="card" style="width: 100%">', f'<div style="width: 100%; height: {height_px}px; background-color: {bg_color};">')
        dark_css = f"""<style>*, *::before, *::after {{ box-sizing: border-box !important; }} html, body {{ background-color: {bg_color} !important; color: #ffffff !important; margin: 0 !important; padding: 0 !important; width: 100% !important; height: 100% !important; overflow: hidden !important; }} .card, .card-body {{ background-color: {bg_color} !important; border: none !important; padding: 0 !important; margin: 0 !important; width: 100% !important; height: 100% !important; }} #mynetwork {{ width: 100% !important; height: {height_px}px !important; background-color: {bg_color} !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }}</style><script>(function(){{function forwardWheel(e){{try{{if(window.parent&&window.parent!==window){{let dy=e.deltaY;let dx=e.deltaX;if(e.deltaMode===1){{dy*=16;dx*=16;}}else if(e.deltaMode===2){{dy*=window.innerHeight;dx*=window.innerWidth;}}window.parent.scrollBy({{top:dy,left:dx,behavior:'auto'}});}}}}catch(err){{}}}}window.addEventListener('wheel',forwardWheel,{{passive:true,capture:true}});document.addEventListener('wheel',forwardWheel,{{passive:true,capture:true}});}})();</script>"""
        return raw_html.replace("</head>", dark_css + "</head>") if "</head>" in raw_html else dark_css + raw_html

def render_concordance_network(cluster_results, has_coll_filter=False, key_suffix=""):
    """
    Renders an interactive Network visualization showing how KWIC node words (or collocates)
    are shared across different sub-corpora restrictions, metadata categories, or clusters.
    """
    st.markdown("### 🕸️ Concordance Network")
    st.markdown(
        "Visualise how KWIC findings (or collocates) are shared across different metadata restrictions, "
        "sub-corpora, or clusters. Shared words will cluster in the centre between the categories."
    )

    if not cluster_results:
        st.info("No concordance dataset available for network visualization.")
        return

    cluster_names = list(cluster_results.keys())
    
    # Global Controls Container
    with st.container(border=True):
        st.markdown("##### ⚙️ Network Configuration")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        
        with c1:
            if has_coll_filter:
                entity_type = st.radio(
                    "Network Entity",
                    ["Node Words (KWIC)", "Collocates"],
                    horizontal=True,
                    key=f"kwic_net_entity_{key_suffix}"
                )
            else:
                entity_type = "Node Words (KWIC)"
                st.caption(f"**Entity**: {entity_type}")

        with c2:
            top_n = st.number_input(
                "Top N Findings per Category",
                min_value=3,
                max_value=250,
                value=10,
                step=5,
                key=f"kwic_net_top_{key_suffix}"
            )

        with c3:
            show_shared_only = st.toggle(
                "Show Only Shared KWIC",
                value=False,
                help="Hides words that appear in only a single category to focus on shared pattern relationships.",
                key=f"kwic_net_shared_only_{key_suffix}"
            )

        with c4:
            base_font_size = st.slider(
                "Node Font Size",
                min_value=14,
                max_value=80,
                value=26,
                step=2,
                key=f"kwic_net_font_{key_suffix}"
            )

        max_cats = len(cluster_names)
        if max_cats > 2:
            min_shared = st.slider(
                "Minimum Shared Categories",
                min_value=2,
                max_value=max_cats,
                value=2,
                disabled=not show_shared_only,
                help="Only show KWIC items shared across at least this number of categories.",
                key=f"kwic_net_min_shared_{key_suffix}"
            )
        else:
            min_shared = 2

    data_by_category = {}
    
    # Single corpus case: check if metadata rows exist to allow grouping
    if len(cluster_names) == 1 and cluster_names[0] in ("Whole Corpus", "Primary", "Corpus"):
        res = cluster_results[cluster_names[0]]
        rows = res.get('rows', [])
        
        meta_keys = set()
        for r in rows:
            meta = r.get('Metadata', {})
            if isinstance(meta, dict):
                meta_keys.update(meta.keys())
                
        if meta_keys:
            selected_meta_key = st.selectbox(
                "Group KWIC Network by Metadata Field:",
                sorted(list(meta_keys)),
                key=f"kwic_net_meta_select_{key_suffix}"
            )
            grouped_counts = {}
            for r in rows:
                meta = r.get('Metadata', {})
                val = str(meta.get(selected_meta_key, "Unspecified")) if isinstance(meta, dict) else "Unspecified"
                if entity_type == "Collocates":
                    item = r.get('Collocate')
                else:
                    item = re.sub(r'<[^>]*>', '', r.get('Node', '')).strip().lower()
                
                if item:
                    if val not in grouped_counts:
                        grouped_counts[val] = {}
                    grouped_counts[val][item] = grouped_counts[val].get(item, 0) + 1
                    
            for cat_val, counts in grouped_counts.items():
                sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                data_by_category[cat_val] = [item for item, cnt in sorted_items]
        else:
            st.warning(
                "⚠️ **Single Corpus View**: To view a Concordance Network across multiple categories, "
                "select categorical metadata filters (e.g. Gender, Review Sentiment) in **Restricted Search** "
                "or run **Cluster Mode**."
            )
            return
    else:
        # Multiple restrictions or cluster mode
        for cat_name in cluster_names:
            res = cluster_results[cat_name]
            if entity_type == "Collocates":
                if 'collocate_counts' in res:
                    counts = res['collocate_counts']
                    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    data_by_category[cat_name] = [item for item, cnt in sorted_items]
                else:
                    rows = res.get('rows', [])
                    counts = {}
                    for r in rows:
                        c = r.get('Collocate')
                        if c: counts[c] = counts.get(c, 0) + 1
                    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    data_by_category[cat_name] = [item for item, cnt in sorted_items]
            else:
                br = res.get('breakdown')
                if br is not None and not br.empty:
                    t_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                    top_words = br.head(top_n)[t_col].tolist()
                    data_by_category[cat_name] = top_words
                else:
                    rows = res.get('rows', [])
                    counts = {}
                    for r in rows:
                        w = re.sub(r'<[^>]*>', '', r.get('Node', '')).strip().lower()
                        if w: counts[w] = counts.get(w, 0) + 1
                    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    data_by_category[cat_name] = [item for item, cnt in sorted_items]

    if not data_by_category or not any(data_by_category.values()):
        st.info("No KWIC data available for network visualization.")
        return

    # Count item sharing
    all_items = []
    for items in data_by_category.values():
        all_items.extend(items)
    
    item_counts = pd.Series(all_items).value_counts().to_dict()

    # Build NetworkX Graph
    G = nx.Graph()

    CATEGORY_COLORS = [
        "#FF6B6B", "#4D96FF", "#6BCB77", "#FFD93D", "#9B5DE5", 
        "#F15BB5", "#00F5D4", "#00BBF9", "#F77F00", "#D62828"
    ]

    # Add category nodes
    for i, cat_name in enumerate(data_by_category.keys()):
        color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
        G.add_node(
            cat_name,
            label=str(cat_name),
            color=color,
            size=45,
            font={'size': base_font_size + 10, 'color': '#ffffff', 'strokeWidth': 4, 'strokeColor': '#000000'},
            shape="dot",
            title=f"Category/Cluster: {cat_name}"
        )

    added_items = set()

    for cat_name, items in data_by_category.items():
        for item in items:
            count = item_counts.get(item, 0)

            if show_shared_only and count < min_shared:
                continue

            if item not in added_items:
                is_shared = count > 1
                node_size = 28 + (count * 4) if is_shared else 20
                node_color = "#FFFF00" if is_shared else "#a5b4fc"
                font_sz = base_font_size + 2 if is_shared else max(12, base_font_size - 10)
                
                G.add_node(
                    item,
                    label=str(item),
                    color=node_color,
                    size=node_size,
                    font={'size': font_sz, 'color': '#ffffff', 'strokeWidth': 3 if is_shared else 2, 'strokeColor': '#000000'},
                    shape="dot",
                    title=f"KWIC Finding: {item}\nShared by {count} categories"
                )
                added_items.add(item)

            edge_width = 4 if count > 1 else 2
            edge_color = "#FFFF00" if count > 1 else "rgba(165, 180, 252, 0.4)"
            G.add_edge(cat_name, item, width=edge_width, color=edge_color)

    # Clean up isolated nodes
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)

    if len(G.nodes) == 0:
        st.info("The network is empty. Try toggling off 'Show Only Shared KWIC' or reducing 'Minimum Shared Categories'.")
        return

    # Render Pyvis
    with st.spinner("Generating Concordance Network..."):
        net = Network(
            height="650px", 
            width="100%", 
            bgcolor="#0f172a", 
            font_color="#ffffff", 
            notebook=False,
            cdn_resources="in_line"
        )
        net.from_nx(G)
        
        physics_json = f"""
        {{
          "nodes": {{ "borderWidth": 2, "font": {{ "size": {base_font_size} }} }},
          "edges": {{ "smooth": {{ "type": "dynamic" }} }},
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
          }},
          "interaction": {{
            "hover": true,
            "navigationButtons": true,
            "zoomView": false,
            "dragNodes": true,
            "dragView": true
          }}
        }}
        """
        net.set_options(physics_json)

        try:
            html_content = prepare_standalone_pyvis_html(net, height_px=650, bg_color="#0f172a")
            st.components.v1.html(html_content, height=670, scrolling=False)
            
        except Exception as e:
            st.error(f"Failed to render pyvis network: {e}")


def render_concordance_overlap_overview(cluster_results, has_coll_filter=False, key_suffix=""):
    """
    Renders an Overlap Size Overview visualization showing how KWIC node words (or collocates)
    are shared across metadata restrictions, sub-corpora, or clusters as proportional horizontal bars.
    """
    st.markdown("### 🔀 Overlap Size Overview")
    st.markdown(
        "View how KWIC findings (or collocates) overlap across different metadata restriction combinations "
        "(e.g., Male Positive, Male Negative, Female Positive, Female Negative). "
        "Horizontal bars group shared findings by their exact category combination, with font sizes scaled by frequency."
    )

    if not cluster_results:
        st.info("No concordance dataset available for overlap overview.")
        return

    cluster_names = list(cluster_results.keys())

    # Global Controls Container
    with st.container(border=True):
        st.markdown("##### ⚙️ Overlap Overview Configuration")
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            if has_coll_filter:
                entity_type = st.radio(
                    "Target Entity",
                    ["Node Words (KWIC)", "Collocates"],
                    horizontal=True,
                    key=f"kwic_ov_entity_{key_suffix}"
                )
            else:
                entity_type = "Node Words (KWIC)"
                st.caption(f"**Entity**: {entity_type}")

        with c2:
            top_n = st.number_input(
                "Top N Findings per Category",
                min_value=3,
                max_value=250,
                value=15,
                step=5,
                key=f"kwic_ov_top_{key_suffix}"
            )

        with c3:
            show_shared_only = st.toggle(
                "Show Only Shared Findings",
                value=True,
                help="Only display words shared across 2 or more metadata restriction categories.",
                key=f"kwic_ov_shared_only_{key_suffix}"
            )

        max_cats = len(cluster_names)
        if max_cats > 2:
            min_shared = st.slider(
                "Minimum Shared Categories",
                min_value=2,
                max_value=max_cats,
                value=2,
                disabled=not show_shared_only,
                help="Only show items shared across at least this number of categories.",
                key=f"kwic_ov_min_shared_{key_suffix}"
            )
        else:
            min_shared = 2

    # Data collection per category
    category_item_counts = {}

    # Single corpus case: check if metadata rows exist to allow grouping
    if len(cluster_names) == 1 and cluster_names[0] in ("Whole Corpus", "Primary", "Corpus"):
        res = cluster_results[cluster_names[0]]
        rows = res.get('rows', [])

        meta_keys = set()
        for r in rows:
            meta = r.get('Metadata', {})
            if isinstance(meta, dict):
                meta_keys.update(meta.keys())

        if meta_keys:
            selected_meta_key = st.selectbox(
                "Group KWIC Overlap by Metadata Field:",
                sorted(list(meta_keys)),
                key=f"kwic_ov_meta_select_{key_suffix}"
            )
            grouped_counts = {}
            for r in rows:
                meta = r.get('Metadata', {})
                val = str(meta.get(selected_meta_key, "Unspecified")) if isinstance(meta, dict) else "Unspecified"
                if entity_type == "Collocates":
                    item = r.get('Collocate')
                else:
                    item = re.sub(r'<[^>]*>', '', r.get('Node', '')).strip().lower()

                if item:
                    if val not in grouped_counts:
                        grouped_counts[val] = {}
                    grouped_counts[val][item] = grouped_counts[val].get(item, 0) + 1

            for cat_val, counts in grouped_counts.items():
                sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
                category_item_counts[cat_val] = dict(sorted_items)
        else:
            st.warning(
                "⚠️ **Single Corpus View**: To view Overlap Size Overview across multiple categories, "
                "select categorical metadata filters in **Restricted Search** or run **Cluster Mode**."
            )
            return
    else:
        # Multiple restrictions or cluster mode
        for cat_name in cluster_names:
            res = cluster_results[cat_name]
            counts = {}
            if entity_type == "Collocates":
                if 'collocate_counts' in res:
                    counts = res['collocate_counts']
                else:
                    rows = res.get('rows', [])
                    for r in rows:
                        c = r.get('Collocate')
                        if c: counts[c] = counts.get(c, 0) + 1
            else:
                br = res.get('breakdown')
                if br is not None and not br.empty:
                    t_col = 'Token Form' if 'Token Form' in br.columns else br.columns[0]
                    f_col = 'Absolute Frequency' if 'Absolute Frequency' in br.columns else (br.columns[1] if len(br.columns)>1 else None)
                    for _, row in br.iterrows():
                        token_str = str(row[t_col]).strip().lower()
                        freq_val = int(row[f_col]) if f_col else 1
                        counts[token_str] = freq_val
                else:
                    rows = res.get('rows', [])
                    for r in rows:
                        w = re.sub(r'<[^>]*>', '', r.get('Node', '')).strip().lower()
                        if w: counts[w] = counts.get(w, 0) + 1

            sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            category_item_counts[cat_name] = dict(sorted_items)

    if not category_item_counts or not any(category_item_counts.values()):
        st.info("No KWIC data available for overlap visualization.")
        return

    # Find which categories each item belongs to
    item_categories = {}
    item_freq_breakdown = {}

    for cat_name, items_dict in category_item_counts.items():
        for item, freq in items_dict.items():
            if item not in item_categories:
                item_categories[item] = set()
                item_freq_breakdown[item] = {}
            item_categories[item].add(cat_name)
            item_freq_breakdown[item][cat_name] = freq

    # Group items by exact combination of categories
    combo_groups = {}
    for item, cats in item_categories.items():
        if show_shared_only and len(cats) < min_shared:
            continue

        combo_key = " & ".join(sorted(list(cats)))
        if combo_key not in combo_groups:
            combo_groups[combo_key] = []

        total_freq = sum(item_freq_breakdown[item].values())
        freq_str = ", ".join([f"{c}: {f}" for c, f in item_freq_breakdown[item].items()])
        combo_groups[combo_key].append((item, total_freq, freq_str))

    # Sort combinations by the number of items descending
    sorted_combos = sorted(combo_groups.items(), key=lambda x: len(x[1]), reverse=True)

    if not sorted_combos:
        st.info("No shared KWIC findings match the current filter criteria. Try unchecking 'Show Only Shared Findings' or reducing 'Minimum Shared Categories'.")
        return

    # Render HTML horizontal bars matching sleek dark UI theme
    html_lines = [
        """
        <style>
        .kwic-overlap-bar-container {
            margin-bottom: 24px;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        .kwic-overlap-label {
            font-weight: 700;
            font-size: 0.95rem;
            color: #f1f5f9;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .kwic-overlap-badge {
            background: rgba(0, 255, 245, 0.15);
            color: #00FFF5;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(0, 255, 245, 0.3);
        }
        .kwic-overlap-bar {
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            border: 1px solid rgba(0, 255, 245, 0.25);
            border-radius: 12px;
            padding: 14px 18px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
            transition: all 0.3s ease;
        }
        .kwic-overlap-bar:hover {
            border-color: rgba(0, 255, 245, 0.6);
            box-shadow: 0 6px 20px rgba(0, 255, 245, 0.12);
        }
        .kwic-overlap-word {
            display: inline-block;
            margin: 6px 12px;
            font-weight: 600;
            cursor: help;
            transition: all 0.2s ease;
        }
        .kwic-overlap-word:hover {
            transform: scale(1.18);
            text-shadow: 0 0 10px rgba(0, 255, 245, 0.6);
        }
        </style>
        """
    ]

    max_count = max([len(items) for combo, items in sorted_combos]) if sorted_combos else 1

    for combo, items in sorted_combos:
        items = sorted(items, key=lambda x: x[1], reverse=True)
        count = len(items)

        width_pct = int(50 + 50 * (count / max_count))

        freqs = [w[1] for w in items]
        min_f = min(freqs) if freqs else 0
        max_f = max(freqs) if freqs else 1
        range_f = max_f - min_f if max_f != min_f else 1

        word_spans = []
        for word, total_freq, freq_detail in items:
            font_size = 13 + 14 * ((total_freq - min_f) / range_f)

            if total_freq > min_f + 0.6 * range_f:
                color = "#00FFF5"
            elif total_freq > min_f + 0.25 * range_f:
                color = "#818CF8"
            else:
                color = "#E2E8F0"

            title_text = f"Token: {word} | Total Freq: {total_freq:,} | Breakdown: [{freq_detail}]"
            word_spans.append(
                f'<span class="kwic-overlap-word" style="font-size: {font_size:.1f}px; color: {color}" title="{title_text}">{word}</span>'
            )

        spans_html = "".join(word_spans)
        html_lines.append(f"""
        <div class="kwic-overlap-bar-container" style="width: {width_pct}%;">
            <div class="kwic-overlap-label">
                <span>🔗 {combo}</span>
                <span class="kwic-overlap-badge">{count} word{"s" if count != 1 else ""}</span>
            </div>
            <div class="kwic-overlap-bar">
                {spans_html}
            </div>
        </div>
        """)

    st.markdown("\n".join(html_lines), unsafe_allow_html=True)
