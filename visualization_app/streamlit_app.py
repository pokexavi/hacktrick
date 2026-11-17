"""Run with: streamlit run visualization_app/streamlit_app.py."""
import streamlit as st

st.set_page_config(page_title="Hacktrick", page_icon=":soccer:", layout="wide")
st.navigation([
    st.Page("pages/data_visualization.py", title="Data visualization", icon=":material/movie:"),
    st.Page("pages/dead_ball.py", title="Dead-ball review", icon=":material/sports:"),
    st.Page("pages/possession.py", title="Possession review", icon=":material/swap_horiz:"),
    st.Page("pages/full_match.py", title="Whole match", icon=":material/timeline:"),
]).run()
