import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import re

def render_concordance_network(cluster_results, has_coll_filter=False, key_suffix=""):
    """
    Renders an interactive Network visualization showing how KWIC node words (or collocates)
    are shared across different sub-corpora restrictions, metadata categories, or clusters.
    """
    st.markdown("---")
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
        c1, c2, c3 = st.columns([2, 1, 1])
        
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

        f1, f2 = st.columns([1, 1])
        with f1:
            min_shared = st.slider(
                "Minimum Shared Categories",
                min_value=2,
                max_value=max(2, len(cluster_names)),
                value=2,
                disabled=not show_shared_only,
                help="Only show KWIC items shared across at least this number of categories.",
                key=f"kwic_net_min_shared_{key_suffix}"
            )
        with f2:
            st.write("")

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
            size=36,
            font={'size': 36, 'color': '#ffffff', 'strokeWidth': 4, 'strokeColor': '#000000'},
            shape="dot",
            title=f"Category/Cluster: {cat_name}"
        )

    added_items = set()
    edges_to_add = []

    for cat_name, items in data_by_category.items():
        for item in items:
            count = item_counts.get(item, 0)

            if show_shared_only and count < min_shared:
                continue

            if item not in added_items:
                node_size = 14 + (count * 5)
                node_color = "#00FFF5" if count > 1 else "#a5b4fc"
                
                G.add_node(
                    item,
                    label=str(item),
                    color=node_color,
                    size=node_size,
                    font={'size': 30, 'color': '#ffffff', 'strokeWidth': 3, 'strokeColor': '#000000'},
                    shape="dot",
                    title=f"KWIC Finding: {item}\nShared by {count} categories"
                )
                added_items.add(item)

            edges_to_add.append((cat_name, item))

    G.add_edges_from(edges_to_add)

    # Clean up isolated nodes
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)

    if len(G.nodes) == 0:
        st.info("The network is empty. Try toggling off 'Show Only Shared KWIC' or reducing 'Minimum Shared Categories'.")
        return

    # Layout calculation
    pos = nx.spring_layout(G, k=1.4 / (len(G.nodes) ** 0.5) if len(G.nodes) > 0 else 0.25, iterations=50)
    for node, coords in pos.items():
        G.nodes[node]['x'] = float(coords[0] * 750)
        G.nodes[node]['y'] = float(coords[1] * 750)

    # Render Pyvis
    with st.spinner("Generating Concordance Network..."):
        net = Network(
            height="850px", 
            width="100%", 
            bgcolor="#0f172a", 
            font_color="#ffffff", 
            notebook=False
        )
        net.from_nx(G)
        
        physics_json = """
        {
          "physics": { "enabled": false },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "zoomView": true
          },
          "edges": {
            "color": {
              "color": "rgba(255, 255, 255, 0.2)",
              "hover": "rgba(0, 255, 245, 0.8)",
              "highlight": "rgba(0, 255, 245, 0.8)"
            },
            "width": 1.5,
            "smooth": { "type": "continuous" }
          }
        }
        """
        net.set_options(physics_json)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                tmp_path = tmp.name
            net.write_html(tmp_path)
            
            with open(tmp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
            html_content = html_content.replace("background-color: #ffffff;", "background-color: #0f172a;")
            html_content = html_content.replace("border: 1px solid lightgray;", "border: 1px solid rgba(255, 255, 255, 0.1);")
            
            st.components.v1.html(html_content, height=870, scrolling=False)
            
        except Exception as e:
            st.error(f"Failed to render pyvis network: {e}")
