import streamlit as st
import uuid

extra_cols = ['alpha', 'beta', 'gamma']

if 'kwic_custom_tiers' not in st.session_state:
    st.session_state['kwic_custom_tiers'] = []

for i, tier in enumerate(st.session_state['kwic_custom_tiers']):
    if 'id' not in tier:
        tier['id'] = str(uuid.uuid4())
    t_id = tier['id']
    
    col_key = f"tier_col_{t_id}"
    align_key = f"tier_align_{t_id}"
    
    if col_key not in st.session_state:
        st.session_state[col_key] = tier.get('col', extra_cols[0])
    if align_key not in st.session_state:
        st.session_state[align_key] = tier.get('align', 'word')
        
    c1, c2, c3, c4 = st.columns([1.5, 4, 3, 1])
    c1.markdown(f"<b>Tier {i+1}</b>", unsafe_allow_html=True)
    
    c2.selectbox("Column", options=extra_cols, key=col_key)
    c3.selectbox("Alignment", options=["word", "sentence"], key=align_key)
    
    if c4.button("❌", key=f"tier_del_{t_id}"):
        st.session_state['kwic_custom_tiers'].pop(i)
        if col_key in st.session_state: del st.session_state[col_key]
        if align_key in st.session_state: del st.session_state[align_key]
        st.rerun()

if st.button("➕ Add Tier", key="btn_add_tier"):
    st.session_state['kwic_custom_tiers'].append({'id': str(uuid.uuid4()), 'col': extra_cols[0], 'align': 'word'})
    st.rerun()

st.write('---')
st.write('Output:')
for tier in st.session_state.get('kwic_custom_tiers', []):
    t_id = tier.get('id')
    col = st.session_state.get(f"tier_col_{t_id}", tier.get('col'))
    align = st.session_state.get(f"tier_align_{t_id}", tier.get('align'))
    st.write(f'{col} - {align}')
