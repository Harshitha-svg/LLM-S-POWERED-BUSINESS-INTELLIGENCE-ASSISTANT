import streamlit as st

def show_preview(df):
    st.dataframe(df.head(50), use_container_width=True)
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
