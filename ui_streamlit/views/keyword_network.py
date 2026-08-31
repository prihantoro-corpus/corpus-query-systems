import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os

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
        c1, c2, c3 = st.columns([2, 1, 1])
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
            show_shared_only, min_shared, key_suffix=f"pos_{key_suffix}"
        )

    with net_tab_neg:
        st.caption("ℹ️ Negative keywords are words used significantly **less** (or missing) in the target compared to the reference.")
        _build_and_render_network(
            res, data_dict, "Negative", top_n, include_overall, 
            show_shared_only, min_shared, key_suffix=f"neg_{key_suffix}"
        )

    with net_tab_comp:
        st.caption("ℹ️ Comparative/Stable words are those that occur with comparable frequencies in both corpora.")
        _build_and_render_network(
            res, data_dict, "Stable", top_n, include_overall, 
            show_shared_only, min_shared, key_suffix=f"comp_{key_suffix}"
        )


def _build_and_render_network(res, data_dict, kw_type, top_n, include_overall, show_shared_only, min_shared, key_suffix):
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
        # Separate color for overall node
        if cat_name == "Overall":
            color = "#E2E8F0"  # Silver/White for central overall node
            size = 70
        else:
            color = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            size = 56
            
        G.add_node(
            cat_name,
            label=str(cat_name),
            color=color,
            size=size,
            font={'size': 30, 'color': '#ffffff', 'strokeWidth': 4, 'strokeColor': '#000000'},
            shape="dot",
            title=f"Category: {cat_name}"
        )

    # Filter and add keyword nodes and edges
    added_keywords = set()
    edges_to_add = []

    for cat_name, words in keywords_by_category.items():
        for word in words:
            count = word_counts.get(word, 0)

            # Apply filters
            if show_shared_only:
                if count < min_shared:
                    continue

            if word not in added_keywords:
                # Size word nodes proportional to how much they are shared
                node_size = 22 + (count * 8)
                node_color = "#00FFF5" if count > 1 else "#a5b4fc"
                
                G.add_node(
                    word,
                    label=str(word),
                    color=node_color,
                    size=node_size,
                    font={'size': 22, 'color': '#ffffff'},
                    shape="dot",
                    title=f"Keyword: {word}\nShared by {count} categories"
                )
                added_keywords.add(word)

            edges_to_add.append((cat_name, word))

    G.add_edges_from(edges_to_add)

    # Clean up categories that have no connected keywords
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(isolated_nodes)

    if len(G.nodes) == 0:
        st.info("The network is empty. Try toggling off 'Show Only Shared Keywords' or reducing 'Minimum Shared Categories'.")
        return

    # Pre-calculate positions using networkx spring layout for static presentation without initial movement
    pos = nx.spring_layout(G, k=1.8 / (len(G.nodes) ** 0.5) if len(G.nodes) > 0 else 0.3, iterations=50)
    for node, coords in pos.items():
        G.nodes[node]['x'] = float(coords[0] * 1000)
        G.nodes[node]['y'] = float(coords[1] * 1000)

    # Render using Pyvis
    with st.spinner("Generating network visualization..."):
        net = Network(
            height="1200px", 
            width="100%", 
            bgcolor="#0f172a", 
            font_color="#ffffff", 
            notebook=False
        )
        net.from_nx(G)
        
        physics_json = """
        {
          "physics": {
            "enabled": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "zoomView": true
          },
          "edges": {
            "color": {
              "color": "rgba(255, 255, 255, 0.18)",
              "hover": "rgba(0, 255, 245, 0.8)",
              "highlight": "rgba(0, 255, 245, 0.8)"
            },
            "width": 1.2,
            "smooth": {
              "type": "continuous"
            }
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
                
            # Replace white background styles from pyvis default template to match CORTEX dark mode
            html_content = html_content.replace(
                "background-color: #ffffff;",
                "background-color: #0f172a;"
            )
            html_content = html_content.replace(
                "border: 1px solid lightgray;",
                "border: 1px solid rgba(255, 255, 255, 0.1);"
            )
            
            st.components.v1.html(html_content, height=1240, scrolling=False)
            
        except Exception as e:
            st.error(f"Failed to render pyvis network: {e}")
