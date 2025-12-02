# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from io import BytesIO
import tempfile 
import os       
import re       

# Import for Pyvis Network Graph
from pyvis.network import Network
import streamlit.components.v1 as components 

# Import for Coordinate Chart
import altair as alt

st.set_page_config(page_title="Corpus Explorer Version 12 Dec 25", layout="wide")

# ---------------------------
# Helpers: stats, IO utilities
# ---------------------------
EPS = 1e-12

def safe_log(x):
    return math.log(max(x, EPS))

def compute_ll(k11, k12, k21, k22):
    """Computes the Log-Likelihood (LL) statistic."""
    total = k11 + k12 + k21 + k22
    if total == 0:
        return 0.0
    
    e11 = (k11 + k12) * (k11 + k21) / total
    e12 = (k11 + k12) * (k12 + k22) / total
    e21 = (k21 + k22) * (k11 + k21) / total
    e22 = (k21 + k22) * (k12 + k22) / total
    
    s = 0.0
    for k,e in ((k11,e11),(k12,e12),(k21,e21),(k22,e22)):
        if k > 0 and e > 0:
            s += k * math.log(k / e)
            
    return 2.0 * s

def compute_mi(k11, target_freq, coll_total, corpus_size):
    """Computes the Mutual Information (MI) statistic."""
    expected = (target_freq * coll_total) / corpus_size
    if expected == 0 or k11 == 0:
        return 0.0
    return math.log2(k11 / expected)

def significance_from_ll(ll_val):
    """Converts Log-Likelihood value to significance level."""
    if ll_val >= 15.13:
        return '*** (p<0.001)'
    if ll_val >= 10.83:
        return '** (p<0.01)'
    if ll_val >= 3.84:
        return '* (p<0.05)'
    return 'ns'

def df_to_excel_bytes(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.getvalue()

@st.cache_data
def create_pyvis_graph(target_word, coll_df):
    """
    Creates a Pyvis interactive network graph. 
    All collocate nodes are Gray due to the generic '##' POS tag.
    """
    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local')
    
    max_ll = coll_df['LL'].max()
    min_ll = coll_df['LL'].min()
    ll_range = max_ll - min_ll
    
    # JSON MUST NOT contain // comments
    net.set_options(f"""
    var options = {{
      "nodes": {{
        "borderWidth": 2,
        "size": 15,
        "font": {{
            "size": 30
        }}
      }},
      "edges": {{
        "width": 5,
        "smooth": {{
          "type": "dynamic"
        }}
      }},
      "physics": {{
        "barnesHut": {{
          "gravitationalConstant": -10000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.9,
          "avoidOverlap": 0.5
        }},
        "minVelocity": 0.75
      }}
    }}
    """)
    
    # 1. Add Target Node 
    net.add_node(target_word, label=target_word, size=40, color='#FFFF00', title=f"Target: {target_word}", font={'color': 'black'})
    
    # Define colors (only Gray is truly relevant now)
    pos_colors = {
        '#': '#AAAAAA',  # Nonsense Tag / Other (Gray)
        'O': '#AAAAAA'   # Other (Gray)
    }
    
    # 2. Add Collocate Nodes and Edges
    for _, row in coll_df.iterrows():
        collocate = row['Collocate']
        ll_score = row['LL']
        observed = row['Observed']
        
        # All nodes default to Gray ('#')
        pos_code = '#' 
        color = pos_colors.get(pos_code, pos_colors['O'])
        
        # Node size based on LL score
        if ll_range > 0:
            normalized_ll = (ll_score - min_ll) / ll_range
            node_size = 15 + normalized_ll * 25 
        else:
            node_size = 25
            
        net.add_node(collocate, label=collocate, size=node_size, color=color, title=f"POS: {row['POS']}\nObs: {observed}\nLL: {ll_score:.2f}")
        net.add_edge(target_word, collocate, value=ll_score, width=5, title=f"LL: {ll_score:.2f}")

    # --- Use a Temporary HTML File ---
    html_content = ""
    temp_path = None
    try:
        temp_filename = "pyvis_graph.html"
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, temp_filename)
        net.write_html(temp_path, notebook=False)
        with open(temp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return html_content
    # --- END OF FIX ---

@st.cache_data
def create_coordinate_chart(chart_df):
    """
    Creates an Altair Coordinate Chart (Upside-Down T).
    Vertical (Y): LL Score. Horizontal (X): Signed Observed Frequency.
    FIX: Ensure explicit encoding for all layers to avoid caching/layering conflicts.
    """
    
    # Max absolute value for a symmetric X-axis
    max_x = chart_df['Signed_Observed'].abs().max() + 1
    max_y = chart_df['LL'].max() * 1.05

    # 1. Define Base Chart and Shared Tooltips
    # Defining X and Y here makes it explicit for all subsequent layers.
    base = alt.Chart(chart_df).encode(
        y=alt.Y('LL', scale=alt.Scale(domain=[0, max_y]), title="Log-Likelihood (Strength)"),
        x=alt.X('Signed_Observed', 
                scale=alt.Scale(domain=[-max_x, max_x]), 
                title="Observed Frequency (Left = Negative | Right = Positive)"),
        tooltip=[
            alt.Tooltip('Collocate', title="Collocate"),
            alt.Tooltip('LL', title="LL Score", format=".2f"),
            alt.Tooltip('Observed', title="Total Observed"),
            alt.Tooltip('Signed_Observed', title="Signed Observed", format=".0f"),
            alt.Tooltip('POS', title="POS")
        ]
    ).properties(
        title="Collocation Strength vs. Positional Observed Frequency"
    )

    # 2. Scatter Points
    points = base.mark_circle(size=60).encode(
        color=alt.condition(
            alt.datum.Signed_Observed < 0,
            alt.value("#FF33B5"),  # Pink for Left Side
            alt.value("#3366FF")   # Blue for Right Side
        )
    )

    # 3. Text Labels (simplified encoding and positioning)
    labels = base.mark_text(
        fontSize=10,
    ).encode(
        text=alt.Text('Collocate'),
        color=alt.condition(
            alt.datum.Signed_Observed < 0,
            alt.value("#FF33B5"),
            alt.value("#3366FF")
        ),
        # Positioning: Use conditional alignment and offset to place the label next to the dot.
        align=alt.condition(alt.datum.Signed_Observed < 0, alt.value('right'), alt.value('left')),
        dx=alt.condition(alt.datum.Signed_Observed < 0, alt.value(-8), alt.value(8))
    )
    
    # 4. Center Line (Upside-down T base)
    center_line = alt.Chart(pd.DataFrame({'x': [0], 'y': [0]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(x='x')

    chart = (points + labels + center_line).interactive()
    
    # Save chart as JSON
    chart_filename = "coordinate_chart.json"
    chart.save(chart_filename)
    return chart_filename


# ---------------------------
# Cached loading (Fast regex tokenization)
# ---------------------------
@st.cache_data
def load_corpus_file(file_bytes, sep=r"\s+"):
    # 1. Try to read as structured 3-column data (Vertical Corpus)
    try:
        file_bytes.seek(0)
        df_attempt = pd.read_csv(file_bytes, sep=sep, header=None, engine="python", dtype=str)
        
        is_vertical = False
        if df_attempt.shape[1] >= 3:
            df_check = df_attempt.iloc[:, :3].copy()
            if df_check.iloc[:, 0].nunique() < len(df_check) * 0.95 and len(df_check) > 5:
                is_vertical = True
            
        if is_vertical:
            df = df_attempt.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
            df["token"] = df["token"].fillna("").astype(str)
            df["pos"] = df["pos"].fillna("###").astype(str)
            df["lemma"] = df["lemma"].fillna("###").astype(str)
            df["_token_low"] = df["token"].str.lower()
            st.sidebar.info("File loaded as **pre-tagged vertical corpus**.")
            return df
            
        file_bytes.seek(0)

    except Exception:
         file_bytes.seek(0) 
         pass

    # 2. Fallback: Treat as Raw Horizontal Text (Fast Regex Tokenizer + Nonsense Tags)
    try:
        raw_text = file_bytes.read().decode('utf-8')
        
        # Regex to split tokens based on space and punctuation, keeping punctuation as separate tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]+', raw_text)
        tokens = [t.strip() for t in tokens if t.strip()] # Clean up any empty strings

        # Assign nonsense tag and lemma
        nonsense_tag = "##"
        nonsense_lemma = "##"
        
        pos_tags = [nonsense_tag] * len(tokens)
        lemmas = [nonsense_lemma] * len(tokens)
        
        df = pd.DataFrame({
            "token": tokens,
            "pos": pos_tags,
            "lemma": lemmas
        })
        
        df["_token_low"] = df["token"].str.lower()
        st.sidebar.warning("File treated as raw text, tagged with **'##'** for fast analysis.")
        return df
        
    except Exception as raw_e:
        st.error(f"Error processing raw text: {raw_e}")
        raise raw_e


# ---------------------------
# UI: header
# ---------------------------
st.title("CORTEX -- Corpus Texts Explorer version 12-Dec-25 (Fast Mode)")
st.caption("Upload vertical corpus (**token POS lemma**) or **raw horizontal text**. Raw text is analyzed quickly using basic tokenization and generic tags (`##`).")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
with st.sidebar:
    st.header("Upload & Options")
    uploaded_file = st.file_uploader("Upload corpus file", type=["txt","csv"])
    st.markdown("---")
    
    # 1. Collocation window control
    coll_window = st.number_input("Collocation window (tokens each side)", min_value=1, max_value=10, value=5, step=1, help="Window used for collocation counting (default ±5).")
    st.markdown("---")
    
    # 2. Concordance context controls (New Widgets)
    st.subheader("KWIC Context (Display)")
    kwic_left = st.number_input("Left Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the left of the node word.")
    kwic_right = st.number_input("Right Context (tokens)", min_value=1, max_value=20, value=7, step=1, help="Number of tokens shown to the right of the node word.")
    st.markdown("---")
    
    # 3. MI min frequency control
    st.write("MI minimum observed frequency (for MI tables).")
    mi_min_freq = st.number_input("MI min observed freq", min_value=1, max_value=100, value=1, step=1)
    st.markdown("---")
    st.write("Shareable deployment tip:")
    st.info("Deploy this app on Streamlit Cloud or HuggingFace Spaces for free sharing.")

if uploaded_file is None:
    st.info("Please upload a corpus file (vertical: token pos lemma, or raw text).")
    st.stop()

# load corpus (cached)
df = load_corpus_file(uploaded_file)
total_tokens = len(df)

# corpus info and frequency list
tokens_lower = df["_token_low"].tolist()
pos_tags = df["pos"].tolist()
PUNCTUATION = {'.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '"', "'", '---', '--', '-', '...', '«', '»', '—'}
tokens_lower_filtered = [t for t in tokens_lower if t not in PUNCTUATION and not t.isdigit()]

token_counts = Counter(tokens_lower_filtered)
unique_types = len(set(tokens_lower_filtered))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"

# STTR per 1000
def compute_sttr_tokens(tokens_list, chunk=1000):
    if len(tokens_list) < chunk:
        return (len(set(tokens_list)) / len(tokens_list)) if len(tokens_list) > 0 else 0.0
    ttrs = []
    for i in range(0, len(tokens_list), chunk):
        c = tokens_list[i:i+chunk]
        if not c: continue
        ttrs.append(len(set(c)) / len(c))
    
    return (sum(ttrs)/len(ttrs)) if ttrs else 0.0

sttr_score = compute_sttr_tokens(tokens_lower_filtered)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Corpus summary")
    info_df = pd.DataFrame({
        "Metric": ["Corpus size (tokens)", "Unique types (w/o punc)", "Lemma count", "STTR (w/o punc, chunk=1000)"],
        "Value": [f"{total_tokens:,}", unique_types, unique_lemmas, round(sttr_score,4)]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True) 

with col2:
    st.subheader("Top frequency (token / POS / freq) (Punctuation skipped)")
    
    freq_df_filtered = df[~df['_token_low'].isin(PUNCTUATION) & ~df['_token_low'].str.isdigit()].copy()
    if all(freq_df_filtered['pos'].str.contains('##')):
         freq_df_filtered['pos'] = '##'

    freq_df = freq_df_filtered.groupby(["token","pos"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)
    
    freq_head = freq_df.head(10).copy()
    freq_head.insert(0,"No", range(1, len(freq_head)+1))
    st.dataframe(freq_head, use_container_width=True, hide_index=True) 
    
    st.download_button("⬇ Download full frequency list (xlsx)", data=df_to_excel_bytes(freq_df), file_name="full_frequency_list_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")

# ---------------------------
# Target input 
# ---------------------------
st.subheader("Target word input")
col_a, col_b = st.columns(2)
with col_a:
    typed_target = st.text_input("Type a token (case-insensitive)", value="")
with col_b:
    uploaded_targets = st.file_uploader("Or upload list of tokens (one per line)", type=["txt","csv"], key="targets_upload")

selected_target = None
if uploaded_targets is not None:
    try:
        target_list = pd.read_csv(uploaded_targets, header=None, squeeze=True, engine="python")[0].astype(str).str.strip().tolist()
    except Exception:
        uploaded_targets.seek(0)
        target_list = uploaded_targets.read().decode('utf-8').splitlines()
        target_list = [t.strip() for t in target_list if t.strip()]
    if target_list:
        selected_target = st.selectbox("Select target from uploaded list", options=target_list)
target_input = (selected_target if selected_target else typed_target).strip()
if not target_input:
    st.info("Type a token or upload a target list and choose one. Then click Analyze.")
analyze_btn = st.button("🔎 Analyze")

# ---------------------------
# Main analysis
# ---------------------------
if analyze_btn and target_input:
    target = target_input.lower()
    positions = [i for i, t in enumerate(tokens_lower) if t == target]
    freq = len(positions)
    if freq == 0:
        st.warning(f"Token '{target_input}' not found in corpus.")
    else:
        st.success(f"Found {freq} occurrences of '{target_input}' (case-insensitive).")
        rel_freq = (freq / total_tokens) * 1_000_000
        st.write(f"Relative frequency: **{rel_freq:.2f}** per million")

        # ---------- CONCORDANCE: KWIC ----------
        st.subheader(f"Concordance (KWIC) — top 10 ({kwic_left} left & {kwic_right} right)")
        kwic_rows = []
        for i in positions:
            left = tokens_lower[max(0, i - kwic_left):i]
            right = tokens_lower[i + 1:i + 1 + kwic_right]
            node_orig = df["token"].iloc[i]
            kwic_rows.append({"Left": " ".join(left), "Node": node_orig, "Right": " ".join(right)})
        kwic_df = pd.DataFrame(kwic_rows)
        kwic_preview = kwic_df.head(10).copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        st.dataframe(kwic_preview, use_container_width=True, hide_index=True)

        st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_df), file_name=f"{target}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ---------- COLLATION: compute stats (MODIFIED for L/R tracking) ----------
        coll_records = []
        for i in positions:
            # Left side
            for j in range(max(0, i - coll_window), i):
                coll_records.append((tokens_lower[j], df["pos"].iloc[j], 'Left'))
            # Right side
            for j in range(i + 1, min(total_tokens, i + coll_window + 1)):
                coll_records.append((tokens_lower[j], df["pos"].iloc[j], 'Right'))

        coll_df_raw = pd.DataFrame(coll_records, columns=["collocate", "pos", "side"])
        
        # Aggregate counts by collocate, pos, AND side
        coll_counts_by_side = coll_df_raw.groupby(["collocate", "pos", "side"]).size().reset_index(name="Observed_Side")

        # Pivot to get L/R counts side-by-side
        coll_counts_pivot = coll_counts_by_side.pivot_table(
            index=["collocate", "pos"], 
            columns="side", 
            values="Observed_Side", 
            fill_value=0
        ).reset_index().rename(columns={'Left': 'Observed_Left', 'Right': 'Observed_Right'})
        
        # Calculate Total Observed and Signed Observed Frequency
        coll_counts_pivot['Observed'] = coll_counts_pivot['Observed_Left'] + coll_counts_pivot['Observed_Right']
        
        coll_counts_pivot['Dominant_Side'] = np.where(coll_counts_pivot['Observed_Left'] > coll_counts_pivot['Observed_Right'], 
                                                      'Left', 'Right')
        # Use total observed magnitude, signed by the dominant side
        coll_counts_pivot['Signed_Observed'] = np.where(coll_counts_pivot['Dominant_Side'] == 'Left', 
                                                       -coll_counts_pivot['Observed'], 
                                                       coll_counts_pivot['Observed'])
        

        stats_list = []
        token_counts_unfiltered = Counter(tokens_lower) 
        
        for _, row in coll_counts_pivot.iterrows():
            w = row["collocate"]
            p = row["pos"]
            observed = int(row["Observed"])
            observed_left = int(row["Observed_Left"])
            observed_right = int(row["Observed_Right"])
            signed_observed = int(row["Signed_Observed"])

            total_freq = token_counts_unfiltered.get(w, 0)
            k11 = observed
            k12 = freq - k11
            k21 = total_freq - k11
            k22 = total_tokens - (k11 + k12 + k21)
            ll = compute_ll(k11, k12, k21, k22)
            mi = compute_mi(k11, freq, total_freq, total_tokens)
            sig = significance_from_ll(ll)
            stats_list.append({
                "Collocate": w,
                "POS": p,
                "Observed": observed,
                "Observed_Left": observed_left,
                "Observed_Right": observed_right,
                "Signed_Observed": signed_observed,
                "LL": round(ll,6),
                "MI": round(mi,6),
                "Significance": sig
            })

        stats_df = pd.DataFrame(stats_list)
        if stats_df.empty:
            st.warning("No collocates found.")
        else:
            stats_df_sorted = stats_df.sort_values("LL", ascending=False)
            network_df = stats_df_sorted.head(20).copy()
            
            chart_df = stats_df_sorted[stats_df_sorted['LL'] > 0].head(50).copy()

            # full LL top10
            full_ll = stats_df_sorted.head(10).copy()
            full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
            
            # full MI (apply MI min freq)
            full_mi_all = stats_df[stats_df["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
            full_mi = full_mi_all.head(10).copy()
            full_mi.insert(0, "Rank", range(1, len(full_mi)+1))

            # --- Coordinate Chart Visualization ---
            st.markdown("---")
            st.subheader("Collocation Positional Bias Coordinate Chart")
            
            if not chart_df.empty:
                chart_filename = create_coordinate_chart(chart_df)
                st.altair_chart(alt.load(chart_filename), use_container_width=True)
                st.caption("This chart displays the **collocation strength (LL)** vertically against the **total observed frequency** horizontally, signed by the dominant position (Left = Pink/Negative, Right = Blue/Positive).")
            else:
                 st.info("Not enough data points with LL > 0 to render the coordinate chart.")


            # --- Collocation Network Graph (Visually all nodes will be Gray) ---
            st.markdown("---")
            st.subheader(f"Interactive Collocation Network (Top {len(network_df)} LL)")
            
            network_html = create_pyvis_graph(target_input, network_df)
            components.html(network_html, height=450)
            
            st.markdown(
                """
                **Graph Key (Simplified):**
                * Central Node (Target): **Yellow**
                * Collocate Node Color: **All nodes are Gray** as POS tags are generic (`##`).
                * Edge Thickness: **All Thick** (Uniform)
                * Collocate Bubble Size: Scales with Log-Likelihood (LL) score (**Bigger Bubble** = Stronger Collocation)
                * Hover over nodes for details (POS: `##`, Observed Freq, LL score).
                """
            )
            st.markdown("---")

            # ---------- Full Collocation Tables (No Tag Filtering) ----------
            st.subheader("Top 10 Collocations (All Tags)")
            
            cols_full = st.columns(2, gap="large")
            
            with cols_full[0]:
                st.markdown("**Log-Likelihood (LL)**")
                st.dataframe(full_ll, use_container_width=True, hide_index=True)
            
            with cols_full[1]:
                st.markdown(f"**Mutual Information (MI) (obs ≥ {mi_min_freq})**")
                st.dataframe(full_mi, use_container_width=True, hide_index=True)


            # ---------- Download Buttons ----------
            st.markdown("---")
            st.subheader("Download Full Results")
            
            cols_to_download = [
                "Collocate", "POS", "Observed", "Observed_Left", 
                "Observed_Right", "LL", "MI", "Significance"
            ]
            
            st.download_button(
                "⬇ Download full LL results (xlsx)", 
                data=df_to_excel_bytes(stats_df_sorted[cols_to_download]), 
                file_name=f"{target}_LL_full.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.download_button(
                f"⬇ Download full MI results (obs≥{mi_min_freq}) (xlsx)", 
                data=df_to_excel_bytes(full_mi_all[cols_to_download]), 
                file_name=f"{target}_MI_full_obsge{mi_min_freq}.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.caption("Tip: This fast processing mode uses generic tags (`##`), which means collocation results are word-based only, ignoring grammar.")
