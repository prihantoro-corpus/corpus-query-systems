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
        dark_css = f"""<style>*, *::before, *::after {{ box-sizing: border-box !important; }} html, body {{ background-color: {bg_color} !important; color: #ffffff !important; margin: 0 !important; padding: 0 !important; width: 100% !important; height: 100% !important; overflow: hidden !important; }} .card, .card-body {{ background-color: {bg_color} !important; border: none !important; padding: 0 !important; margin: 0 !important; width: 100% !important; height: 100% !important; }} #mynetwork {{ width: 100% !important; height: {height_px}px !important; background-color: {bg_color} !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }}</style>"""
        return raw_html.replace("</head>", dark_css + "</head>") if "</head>" in raw_html else dark_css + raw_html

def render_keyword_network(res, key_suffix=""):
    """
    Renders interactive network visualizations showing how keywords are shared
    between different sub-corpora or individual files.
    """
    st.markdown("### 🕸️ Keyword Network")
    st.markdown(
        "Visualise how keywords are shared across different domains, sub-corpora attributes, or files. "
        "Shared keywords will cluster in the centre between the categories they belong to."
    )

    by_file = res.get('by_filename', {})
    by_attr = res.get('by_attributes', {})

    # Determine grouping options
    group_options = []
    if by_attr:
        for attr in by_attr.keys():
            group_options.append(f"Sub-corpora Attribute: {attr}")
    if by_file:
        group_options.append("Individual Files")

    if not group_options:
        st.warning(
            "⚠️ **No grouped keyword data available.**\n\n"
            "To use the Keyword Network, please run the keyword calculation again and check "
            "**By Individual File** or **By Sub-corpora Attributes** under the *Analysis Basis* settings."
        )
        return

    # Global Controls for Network
    with st.container(border=True):
        st.markdown("##### ⚙️ Network Configuration")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            selected_group = st.selectbox(
                "Group Network By",
                group_options,
                key=f"kw_net_group_{key_suffix}"
            )
        with c2:
            top_n = st.number_input(
                "Top N Keywords per Category",
                min_value=3,
                max_value=100,
                value=10,
                key=f"kw_net_top_{key_suffix}"
            )
        with c3:
            hide_overall = st.checkbox(
                "Hide 'Overall' Category",
                value=False,
                help="Hide the overall corpus keywords from the network visualization.",
                key=f"kw_net_hide_overall_{key_suffix}"
            )
            include_overall = not hide_overall
        with c4:
            base_font_size = st.slider(
                "Node Font Size",
                min_value=14,
                max_value=80,
                value=26,
                step=2,
                key=f"kw_net_font_{key_suffix}"
            )

        f1, f2 = st.columns(2)
        with f1:
            show_shared_only = st.toggle(
                "Show Only Shared Keywords",
                value=False,
                help="Hides keywords that are unique to a single category/domain to highlight relationships.",
                key=f"kw_net_shared_only_{key_suffix}"
            )
        with f2:
            # We calculate this dynamically inside the builder but let's provide a slider
            min_shared = st.slider(
                "Minimum Shared Categories",
                min_value=2,
                max_value=10,
                value=2,
                disabled=not show_shared_only,
                help="Only show keywords shared across at least this number of categories.",
                key=f"kw_net_min_shared_{key_suffix}"
            )

    # Determine dataset
    data_dict = {}
    if selected_group == "Individual Files":
        data_dict = by_file
    else:
        attr_name = selected_group.replace("Sub-corpora Attribute: ", "")
        data_dict = by_attr.get(attr_name, {})

    if not data_dict:
        st.info("No data available for the selected grouping.")
        return

    # Create network tabs
    net_tab_pos, net_tab_neg, net_tab_comp = st.tabs([
        "🟢 Positive Keyword Network",
        "🔴 Negative Keyword Network",
        "🔵 Comparative Keyword Network"
    ])

    with net_tab_pos:
        st.caption("ℹ️ Positive keywords are words used significantly **more** in the target than in the reference.")
        _build_and_render_network(
            res, data_dict, "Positive", top_n, include_overall, 
            show_shared_only, min_shared, base_font_size, key_suffix=f"pos_{key_suffix}"
        )

    with net_tab_neg:
        st.caption("ℹ️ Negative keywords are words used significantly **less** (or missing) in the target compared to the reference.")
        _build_and_render_network(
            res, data_dict, "Negative", top_n, include_overall, 
            show_shared_only, min_shared, base_font_size, key_suffix=f"neg_{key_suffix}"
        )

    with net_tab_comp:
        st.caption("ℹ️ Comparative/Stable words are those that occur with comparable frequencies in both corpora.")
        _build_and_render_network(
            res, data_dict, "Stable", top_n, include_overall, 
            show_shared_only, min_shared, base_font_size, key_suffix=f"comp_{key_suffix}"
        )


def _build_and_render_network(res, data_dict, kw_type, top_n, include_overall, show_shared_only, min_shared, base_font_size=26, key_suffix=""):
    # Extract keywords per category
    keywords_by_category = {}

    # 1. Extract Overall if requested
    if include_overall:
        overall_df = res.get('overall')
        if overall_df is not None and not overall_df.empty:
            filtered = overall_df[overall_df['Type'] == kw_type]
            if kw_type == 'Negative':
                filtered = filtered.sort_values('LL', ascending=False)
            elif kw_type == 'Stable':
                filtered = filtered.sort_values('LL', ascending=True)
            else:
                filtered = filtered.sort_values('LL', ascending=False)
            top_words = filtered.head(top_n)['token'].tolist()
            if top_words:
                keywords_by_category["Overall"] = top_words

    # 2. Extract Category-specific keywords
    for cat_name, df in data_dict.items():
        if df is not None and not df.empty:
            filtered = df[df['Type'] == kw_type]
            if kw_type == 'Negative':
                filtered = filtered.sort_values('LL', ascending=False)
            elif kw_type == 'Stable':
                filtered = filtered.sort_values('LL', ascending=True)
            else:
                filtered = filtered.sort_values('LL', ascending=False)
            
            top_words = filtered.head(top_n)['token'].tolist()
            if top_words:
                keywords_by_category[cat_name] = top_words

    if not keywords_by_category:
        st.info(f"No {kw_type.lower()} keywords found to build network.")
        return

    # Count word frequencies/sharing
    all_words = []
    for words in keywords_by_category.values():
        all_words.extend(words)
    
    word_series = pd.Series(all_words)
    word_counts = word_series.value_counts().to_dict()

    # Build NetworkX Graph
    G = nx.Graph()

    # High-contrast color palette for category nodes
    CATEGORY_COLORS = [
        "#FF6B6B", "#4D96FF", "#6BCB77", "#FFD93D", "#9B5DE5", 
        "#F15BB5", "#00F5D4", "#00BBF9", "#F77F00", "#D62828"
    ]

    # Add category nodes
    for i, cat_name in enumerate(keywords_by_category.keys()):
        if cat_name == "Overall":
            color = "#E2E8F0"
            size = 50
        else:
            color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            size = 45
            
        cat_font_sz = base_font_size + 14 if cat_name == "Overall" else base_font_size + 10
        G.add_node(
            cat_name,
            label=str(cat_name),
            color=color,
            size=size,
            font={'size': cat_font_sz, 'color': '#ffffff', 'strokeWidth': 4, 'strokeColor': '#000000'},
            shape="dot",
            title=f"Category: {cat_name}"
        )

    added_keywords = set()

    for cat_name, words in keywords_by_category.items():
        for word in words:
            count = word_counts.get(word, 0)

            if show_shared_only and count < min_shared:
                continue

            if word not in added_keywords:
                is_shared = count > 1
                node_size = 28 + (count * 4) if is_shared else 20
                node_color = "#FFFF00" if is_shared else "#a5b4fc"
                word_font_sz = base_font_size + 2 if is_shared else max(12, base_font_size - 10)
                
                G.add_node(
                    word,
                    label=str(word),
                    color=node_color,
                    size=node_size,
                    font={'size': word_font_sz, 'color': '#ffffff', 'strokeWidth': 3 if is_shared else 2, 'strokeColor': '#000000'},
                    shape="dot",
                    title=f"Keyword: {word}\nShared by {count} categories"
                )
                added_keywords.add(word)

            edge_width = 4 if count > 1 else 2
            edge_color = "#FFFF00" if count > 1 else "rgba(165, 180, 252, 0.4)"
            G.add_edge(cat_name, word, width=edge_width, color=edge_color)

    # Clean up categories that have no connected keywords
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)

    if len(G.nodes) == 0:
        st.info("The network is empty. Try toggling off 'Show Only Shared Keywords' or reducing 'Minimum Shared Categories'.")
        return

    # Render using Pyvis
    with st.spinner("Generating network visualization..."):
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
