# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from io import BytesIO

st.set_page_config(page_title="Corpus Collocation Explorer", layout="wide")

# ---------------------------
# Helpers: stats, IO utilities
# ---------------------------
EPS = 1e-12

def safe_log(x):
    return math.log(max(x, EPS))

def compute_ll(k11, k12, k21, k22):
    """
    Computes the Log-Likelihood (LL) statistic based on a 2x2 contingency table.
    k11: observed frequency of target and collocate
    k12: observed frequency of target but not collocate
    k21: observed frequency of collocate but not target
    k22: observed frequency of neither target nor collocate
    """
    total = k11 + k12 + k21 + k22
    if total == 0:
        return 0.0
    
    # Expected frequencies
    e11 = (k11 + k12) * (k11 + k21) / total
    e12 = (k11 + k12) * (k12 + k22) / total
    e21 = (k21 + k22) * (k11 + k21) / total
    e22 = (k21 + k22) * (k12 + k22) / total
    
    s = 0.0
    # Sum of k * log(k / e) for each cell
    for k,e in ((k11,e11),(k12,e12),(k21,e21),(k22,e22)):
        # Check to avoid log(0)
        if k > 0 and e > 0:
            s += k * math.log(k / e)
            
    # LL = 2 * sum
    return 2.0 * s

def compute_mi(k11, target_freq, coll_total, corpus_size):
    """
    Computes the Mutual Information (MI) statistic.
    k11: observed frequency of target and collocate
    target_freq: frequency of the target word
    coll_total: frequency of the collocate word
    corpus_size: total number of tokens
    """
    expected = (target_freq * coll_total) / corpus_size
    if expected == 0 or k11 == 0:
        return 0.0
    # MI = log2(Observed / Expected)
    return math.log2(k11 / expected)

def significance_from_ll(ll_val):
    """
    Converts Log-Likelihood value to significance level (based on 1 d.f. Chi-squared).
    """
    if ll_val >= 15.13:
        return '*** (p<0.001)'
    if ll_val >= 10.83:
        return '** (p<0.01)'
    if ll_val >= 3.84:
        return '* (p<0.05)'
    return 'ns'

def df_to_excel_bytes(df):
    """
    Converts a pandas DataFrame to an Excel file (xlsx) in memory (BytesIO).
    Fix: Removed writer.save() which is deprecated in modern pandas versions.
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        # writer.save() is no longer needed/supported within the 'with' block
    buf.seek(0)
    return buf.getvalue()

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------------------------
# Cached loading
# ---------------------------
@st.cache_data
def load_corpus_file(file_bytes, sep=r"\s+"):
    # Attempt to read vertical 3-column format token pos lemma
    try:
        # read as whitespace-separated with no header
        df = pd.read_csv(file_bytes, sep=sep, header=None, engine="python", dtype=str)
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
        else:
            # fallback to reading headers
            df = pd.read_csv(file_bytes, engine="python", dtype=str)
            cols = [c.lower() for c in df.columns]
            if 'token' in cols:
                df.columns = [c.lower() for c in df.columns]
            elif 'word' in cols:
                df.rename(columns={df.columns[cols.index('word')]: 'token'}, inplace=True)
            else:
                # assume first three columns are token,pos,lemma
                df = df.iloc[:, :3]
                df.columns = ["token", "pos", "lemma"]
    except Exception as e:
        # last resort: try pandas default
        df = pd.read_csv(file_bytes, engine="python", dtype=str)
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
            df.columns = ["token", "pos", "lemma"]
        else:
            raise e
    # normalize and fill
    df["token"] = df["token"].fillna("").astype(str)
    df["pos"] = df["pos"].fillna("###").astype(str)
    df["lemma"] = df["lemma"].fillna("###").astype(str)
    # create lowercase tokens for matching
    df["_token_low"] = df["token"].str.lower()
    return df

# ---------------------------
# UI: header
# ---------------------------
st.title("Corpus Collocation Explorer")
st.caption("Upload vertical corpus (token POS lemma). Search tokens, view KWIC, LL & MI collocates, and download results.")

# ---------------------------
# Panel: upload and corpus info
# ---------------------------
with st.sidebar:
    st.header("Upload & Options")
    uploaded_file = st.file_uploader("Upload corpus file (vertical: token POS lemma)", type=["txt","csv"])
    st.markdown("---")
    # collocation window fixed as per your choice, but allow user small control if they want
    coll_window = st.number_input("Collocation window (tokens each side)", min_value=1, max_value=10, value=5, step=1, help="Window used for collocation counting (default ±5).")
    st.write("KWIC context is fixed at 7 left / 7 right (display).")
    st.markdown("---")
    st.write("MI minimum observed frequency (for MI tables).")
    mi_min_freq = st.number_input("MI min observed freq", min_value=1, max_value=100, value=1, step=1)
    st.markdown("---")
    st.write("Shareable deployment tip:")
    st.info("Deploy this app on Streamlit Cloud or HuggingFace Spaces for free sharing.")

if uploaded_file is None:
    st.info("Please upload a corpus file (vertical: token pos lemma).")
    st.stop()

# load corpus (cached)
df = load_corpus_file(uploaded_file)
total_tokens = len(df)
st.sidebar.success(f"Loaded: {total_tokens:,} tokens")

# corpus info and frequency list
tokens_lower = df["_token_low"].tolist()
pos_tags = df["pos"].tolist()
token_counts = Counter(tokens_lower)
unique_types = len(set(tokens_lower))
unique_lemmas = df["lemma"].nunique() if "lemma" in df.columns else "###"

# STTR per 1000
def compute_sttr_tokens(tokens_list, chunk=1000):
    if len(tokens_list) < chunk:
        return (len(set(tokens_list)) / len(tokens_list)) * 1000
    ttrs = []
    for i in range(0, len(tokens_list), chunk):
        c = tokens_list[i:i+chunk]
        if not c: continue
        ttrs.append(len(set(c)) / len(c))
    return (sum(ttrs)/len(ttrs))*1000 if ttrs else 0.0

sttr_score = compute_sttr_tokens(tokens_lower)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Corpus summary")
    info_df = pd.DataFrame({
        "Metric": ["Corpus size (tokens)", "Unique types", "Lemma count", "STTR (per 1000)"],
        "Value": [f"{total_tokens:,}", unique_types, unique_lemmas, round(sttr_score,4)]
    })
    st.table(info_df)

with col2:
    st.subheader("Top frequency (token / POS / freq)")
    freq_df = df.groupby(["token","pos"]).size().reset_index(name="frequency").sort_values("frequency", ascending=False).reset_index(drop=True)
    freq_head = freq_df.head(10).copy()
    freq_head.insert(0,"No", range(1, len(freq_head)+1))
    st.table(freq_head)
    # download freq
    st.download_button("⬇ Download full frequency list (xlsx)", data=df_to_excel_bytes(freq_df), file_name="full_frequency_list.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")

# ---------------------------
# Target input (option C: both)
# ---------------------------
st.subheader("Target word input")
col_a, col_b = st.columns(2)
with col_a:
    typed_target = st.text_input("Type a token (case-insensitive)", value="")
with col_b:
    uploaded_targets = st.file_uploader("Or upload list of tokens (one per line)", type=["txt","csv"], key="targets_upload")

# build selectable targets if list uploaded
selected_target = None
if uploaded_targets is not None:
    try:
        # try to read lines
        target_list = pd.read_csv(uploaded_targets, header=None, squeeze=True, engine="python")[0].astype(str).str.strip().tolist()
    except Exception:
        uploaded_targets.seek(0)
        target_list = uploaded_targets.read().decode('utf-8').splitlines()
        target_list = [t.strip() for t in target_list if t.strip()]
    if target_list:
        # create a dropdown so user picks one
        selected_target = st.selectbox("Select target from uploaded list", options=target_list)
# Final chosen target precedence: selected_target > typed_target
target_input = (selected_target if selected_target else typed_target).strip()
if not target_input:
    st.info("Type a token or upload a target list and choose one. Then click Analyze.")
# analyze button
analyze_btn = st.button("🔎 Analyze")

# ---------------------------
# Main analysis
# ---------------------------
if analyze_btn and target_input:
    target = target_input.lower()
    # find positions
    positions = [i for i, t in enumerate(tokens_lower) if t == target]
    freq = len(positions)
    if freq == 0:
        st.warning(f"Token '{target_input}' not found in corpus.")
    else:
        st.success(f"Found {freq} occurrences of '{target_input}' (case-insensitive).")
        rel_freq = (freq / total_tokens) * 1_000_000
        st.write(f"Relative frequency: **{rel_freq:.2f}** per million")

        # ---------- CONCORDANCE: exact 7 left / 7 right (KWIC) ----------
        st.subheader("Concordance (KWIC) — top 10 (7 left & 7 right)")
        kwic_rows = []
        for i in positions:
            left = tokens_lower[max(0, i-7):i]
            right = tokens_lower[i+1:i+1+7]
            # display original-case node
            node_orig = df["token"].iloc[i]
            kwic_rows.append({"Left": " ".join(left), "Node": node_orig, "Right": " ".join(right)})
        kwic_df = pd.DataFrame(kwic_rows)
        kwic_preview = kwic_df.head(10).copy().reset_index(drop=True)
        kwic_preview.insert(0, "No", range(1, len(kwic_preview)+1))
        st.table(kwic_preview)

        # full concordance download
        st.download_button("⬇ Download full concordance (xlsx)", data=df_to_excel_bytes(kwic_df), file_name=f"{target}_full_concordance.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # ---------- COLLATION: collect collocates within ±coll_window ----------
        coll_pairs = []
        for i in positions:
            start = max(0, i - coll_window)
            end = min(total_tokens, i + coll_window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                coll_pairs.append((tokens_lower[j], df["pos"].iloc[j]))

        coll_df = pd.DataFrame(coll_pairs, columns=["collocate", "pos"])
        coll_counts = coll_df.groupby(["collocate","pos"]).size().reset_index(name="Observed")

        # compute stats
        stats_list = []
        for _, row in coll_counts.iterrows():
            w = row["collocate"]
            p = row["pos"]
            observed = int(row["Observed"])
            total_freq = token_counts.get(w, 0)
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
                "Total_Freq": total_freq,
                "LL": round(ll,6),
                "MI": round(mi,6),
                "Significance": sig
            })

        stats_df = pd.DataFrame(stats_list)
        if stats_df.empty:
            st.warning("No collocates found.")
        else:
            # full LL top10
            full_ll = stats_df.sort_values("LL", ascending=False).reset_index(drop=True).head(10).copy()
            full_ll.insert(0, "Rank", range(1, len(full_ll)+1))
            # full MI (apply MI min freq)
            full_mi_all = stats_df[stats_df["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True)
            full_mi = full_mi_all.head(10).copy()
            full_mi.insert(0, "Rank", range(1, len(full_mi)+1))

            # category tables (multi-POS allowed)
            def category_df(prefixes):
                mask = pd.Series(False, index=stats_df.index)
                for pref in prefixes:
                    mask = mask | stats_df["POS"].str.startswith(pref, na=False)
                sub = stats_df[mask].copy()
                ll_sub = sub.sort_values("LL", ascending=False).reset_index(drop=True).head(10).copy()
                ll_sub.insert(0, "Rank", range(1, len(ll_sub)+1))
                mi_sub = sub[sub["Observed"] >= mi_min_freq].sort_values("MI", ascending=False).reset_index(drop=True).head(10).copy()
                mi_sub.insert(0, "Rank", range(1, len(mi_sub)+1))
                return ll_sub, mi_sub

            ll_N, mi_N = category_df(("N",))    # N (NN*, NNP*, ...)
            ll_V, mi_V = category_df(("V",))
            ll_J, mi_J = category_df(("J",))
            ll_R, mi_R = category_df(("R",))

            # ---------- Downloads for full result sets ----------
            st.download_button("⬇ Download full LL (all collocates) (xlsx)", data=df_to_excel_bytes(stats_df.sort_values("LL", ascending=False)), file_name=f"{target}_LL_full.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button(f"⬇ Download full MI (obs≥{mi_min_freq}) (xlsx)", data=df_to_excel_bytes(full_mi_all), file_name=f"{target}_MI_full_obsge{mi_min_freq}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ---------- Side-by-side display for LL: Full | N | V | J | R ----------
            st.subheader("Log-Likelihood — Top 10 (side-by-side)")
            cols = st.columns(5, gap="small")
            with cols[0]:
                st.markdown("**Full (LL)**")
                st.table(full_ll)
            with cols[1]:
                st.markdown("**N (N*) — LL**")
                st.table(ll_N)
            with cols[2]:
                st.markdown("**V (V*) — LL**")
                st.table(ll_V)
            with cols[3]:
                st.markdown("**J (J*) — LL**")
                st.table(ll_J)
            with cols[4]:
                st.markdown("**R (R*) — LL**")
                st.table(ll_R)

            # small per-category download buttons (These are currently all displayed right after the LL tables in the original script)
            # The download buttons are placed out of the columns block so they take full width.
            # I will keep the original placement for the download buttons below, but they are currently slightly cluttered.

            st.markdown("---")
            st.subheader("Download Top 10 Tables")
            
            # This loop prints the buttons inline/scattered because it's outside any column/container.
            # I'll modify the loop to use columns for better organization, replacing the previous `coldl = st.columns(5)` which was unused.
            
            # Group buttons by metric type for clarity
            
            st.markdown("**LL Top 10 Downloads**")
            ll_dl_cols = st.columns(5)
            ll_mapping = {
                "Full": full_ll, "N": ll_N, "V": ll_V, "J": ll_J, "R": ll_R
            }
            for i, (cat, df_tab) in enumerate(ll_mapping.items()):
                bname = f"LL {cat} top10"
                ll_dl_cols[i].download_button(f"⬇ {bname} (xlsx)", data=df_to_excel_bytes(df_tab), file_name=f"{target}_LL_{cat}_top10.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.markdown("---")
            st.markdown(f"**MI Top 10 Downloads (obs ≥ {mi_min_freq})**")
            mi_dl_cols = st.columns(5)
            mi_mapping = {
                "Full": full_mi, "N": mi_N, "V": mi_V, "J": mi_J, "R": mi_R
            }
            for i, (cat, df_tab) in enumerate(mi_mapping.items()):
                bname = f"MI {cat} top10"
                mi_dl_cols[i].download_button(f"⬇ {bname} (xlsx)", data=df_to_excel_bytes(df_tab), file_name=f"{target}_MI_{cat}_top10.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
            st.markdown("---")

            # ---------- Side-by-side display for MI: Full | N | V | J | R ----------
            st.subheader(f"Mutual Information — Top 10 (obs ≥ {mi_min_freq})")
            cols_mi = st.columns(5, gap="small")
            with cols_mi[0]:
                st.markdown("**Full (MI)**")
                st.table(full_mi)
            with cols_mi[1]:
                st.markdown("**N (N*) — MI**")
                st.table(mi_N)
            with cols_mi[2]:
                st.markdown("**V (V*) — MI**")
                st.table(mi_V)
            with cols_mi[3]:
                st.markdown("**J (J*) — MI**")
                st.table(mi_J)
            with cols_mi[4]:
                st.markdown("**R (R*) — MI**")
                st.table(mi_R)

st.caption("Tip: Deploy this file to Streamlit Cloud or HuggingFace Spaces to share with others.")