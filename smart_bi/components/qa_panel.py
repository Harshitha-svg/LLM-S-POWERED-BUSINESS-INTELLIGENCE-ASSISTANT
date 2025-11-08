import streamlit as st
from backend import query_embeddings

def chat_panel():
    query = st.text_input("Ask your business question:")
    if st.button("Ask"):
        results = query_embeddings(query)
        for r in results:
            st.markdown(f"**Result:** {r['metadata']['text']}  \nScore: {r['score']}")

