import streamlit as st

st.header("Tab 2: Future Environment")
st.info("This page will feature a more complex environment with different actions and parameters.")

col_controls, col_spacer, col_main = st.columns([1, 0.1, 5])

with col_controls:
    st.subheader("Controls (Coming Soon)")
    st.write("Independent config will go here")

with col_spacer:
    st.markdown(
        """<div style='border-left: 2px solid #e0e0e0; height: 100vh;'></div>""",
        unsafe_allow_html=True,
    )

with col_main:
    st.write("Environment visualization will go here")

