import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

def page0_main():
    st.markdown("#### Scalable Entity ALignment (SEAL) demo")
    

page_names_to_funcs = {
    "Main Page": page0_main,
    "Train model": None,
    "Query model": None
} 

def main():
    st.set_page_config(page_title="SEAL demo",
                       page_icon = "images/seal.jpg")
    #st.write(css, unsafe_allow_html=True)
    st.session_state.disabled =True 

    with st.sidebar:
        st.sidebar.image("../images/seal2.jpg", width=200)
        selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]() 

    with st.sidebar:
        st.subheader("Uploads")


if __name__ == '__main__':
    main()