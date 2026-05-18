import streamlit as st
import duckdb
from core.preprocessing.xml_parser import get_xml_attribute_columns, is_integer_col

def render_xml_restriction_filters(db_path, view_name, corpus_name=None):
    """
    Dynamically renders UI filters for XML attributes found in the DuckDB corpus.
    Returns a dictionary of selected values: {attribute_name: {'type': 'list/range', ...}}.
    """
    if not db_path:
        return None
        
    con = duckdb.connect(db_path, read_only=True)
    try:
        attr_cols = get_xml_attribute_columns(con)
        
        # Show count in label
        display_name = corpus_name if corpus_name else db_path.split('_')[-1]
        
        if not attr_cols:
            label = f"🎯 Restricted Search (No metadata detected in '{display_name}')"
            with st.expander(label, expanded=False):
                st.info("No XML attributes or custom metadata columns were found in this corpus. Restricted search is only available for corpora with segment-level metadata.")
            return None

        label = f"🎯 Restricted Search ({len(attr_cols)} filters related to '{display_name}')"
        with st.expander(label, expanded=False):
            st.markdown("**Narrow your search by selecting specific XML attribute values.**")
            st.caption("For Numerical attributes (e.g. Year, ID), specify a Range. For Text attributes, select values from the dropdown.")
            
            selected_filters = {}
            num_cols = min(len(attr_cols), 4)
            cols = st.columns(num_cols)
            
            for i, attr in enumerate(attr_cols):
                with cols[i % num_cols]:
                    if is_integer_col(con, attr):
                        try:
                            stats = con.execute(f"SELECT MIN(CAST({attr} AS BIGINT)), MAX(CAST({attr} AS BIGINT)) FROM corpus").fetchone()
                            min_val, max_val = stats[0], stats[1]
                            
                            if min_val is None or max_val is None:
                                continue
                                
                            st.markdown(f"**{attr.capitalize()}**")
                            c1, c2 = st.columns(2)
                            with c1:
                                val_min = st.number_input(f"Min", min_value=min_val, max_value=max_val, value=min_val, key=f"xml_int_min_{attr}_{view_name}")
                            with c2:
                                val_max = st.number_input(f"Max", min_value=min_val, max_value=max_val, value=max_val, key=f"xml_int_max_{attr}_{view_name}")
                            
                            if val_min > min_val or val_max < max_val:
                                selected_filters[attr] = {'type': 'range', 'min': val_min, 'max': val_max}
                            elif val_min > val_max:
                                st.warning("Min > Max")
                        except Exception as e:
                            st.error(f"Error loading {attr}: {e}")
                    else: 
                        try:
                            raw_vals = [r[0] for r in con.execute(f"SELECT DISTINCT {attr} FROM corpus WHERE {attr} IS NOT NULL ORDER BY {attr}").fetchall()]
                            # Clean, strip, and deduplicate values to prevent Streamlit silent selection failures
                            cleaned_vals = [str(v).strip() for v in raw_vals if str(v).strip() and str(v).lower() != 'nan']
                            unique_vals = list(dict.fromkeys(cleaned_vals))
                        except:
                            continue
                        
                        if not unique_vals: continue
                    
                        selected = st.multiselect(
                            f"{attr.capitalize()}", 
                            options=unique_vals, 
                            key=f"xml_filter_{attr}_{view_name}"
                        )
                        if selected:
                            selected_filters[attr] = {'type': 'list', 'values': selected}
            
            if selected_filters:
                st.caption(f"Active restrictions: {len(selected_filters)} attributes filtered.")
            else:
                st.caption("No restrictions applied.")
                
            return selected_filters
    finally:
        con.close()
