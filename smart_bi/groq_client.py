import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

def query_groq(prompt, df=None):
    """
    Sends a question to Groq LLM.
    If a dataframe is provided, context is added for data-aware answers.
    """
    try:
        # Add data context if available
        if df is not None:
            context = f"""
            You are a Business Intelligence Assistant.
            The user has uploaded a dataset with these columns:
            {', '.join(df.columns.tolist())}

            Here are some example rows:
            {df.head(5).to_markdown(index=False)}

            Based on this data, answer the following question concisely and with calculations if relevant:
            {prompt}
            """
        else:
            context = prompt

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": context}],
            "temperature": 0.3
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Error from Groq API: {e}"
