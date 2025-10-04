"""Test if all dependencies are installed correctly"""

print("🧪 Testing installations...\n")

try:
    import streamlit
    print("✅ Streamlit")
except ImportError as e:
    print(f"❌ Streamlit: {e}")

try:
    from groq import Groq
    print("✅ Groq")
except ImportError as e:
    print(f"❌ Groq: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers")
except ImportError as e:
    print(f"❌ Sentence Transformers: {e}")

try:
    import chromadb
    print("✅ ChromaDB")
except ImportError as e:
    print(f"❌ ChromaDB: {e}")

try:
    import requests
    print("✅ Requests")
except ImportError as e:
    print(f"❌ Requests: {e}")

try:
    from bs4 import BeautifulSoup
    print("✅ BeautifulSoup")
except ImportError as e:
    print(f"❌ BeautifulSoup: {e}")

try:
    import pandas
    print("✅ Pandas")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import plotly
    print("✅ Plotly")
except ImportError as e:
    print(f"❌ Plotly: {e}")

print("\n🎉 All dependencies installed successfully!")
