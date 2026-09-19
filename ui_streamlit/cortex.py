import streamlit as st

st.set_page_config(page_title="Redirecting to CORTEX...", layout="centered")

st.warning("🚀 **CORTEX has moved!** For better performance and higher memory limits, this app is now hosted on Hugging Face Spaces.")
st.info("Redirecting you to the new home in 3 seconds... If nothing happens, [**click here**](https://huggingface.co/spaces/prihantoro-corpus/cortex)!")

import streamlit.components.v1 as components
components.html(
    '''
    <script>
        setTimeout(function() {
            window.parent.location.href = "https://huggingface.co/spaces/prihantoro-corpus/cortex";
        }, 3000);
    </script>
    ''',
    height=0
)
st.stop()
