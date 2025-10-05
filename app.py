"""
Product Comparison RAG - Streamlit Application
"""

import streamlit as st
from src.scrapers.jina_reader import JinaReader
from src.extractors.product_extractor import ProductExtractor
from src.rag.vector_store import VectorStore
from src.rag.synthesizer import get_synthesizer
import time
from typing import List


# Page configuration
st.set_page_config(
    page_title="Product Comparison RAG",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .product-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    
    .comparison-result {
        background: #ffffff;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'comparison_history' not in st.session_state:
    st.session_state.comparison_history = []


def add_products(urls: List[str]):
    """Add products from URLs"""
    
    if not urls:
        st.warning("⚠️ Please enter at least one URL")
        return
    
    with st.spinner("🔄 Processing products..."):
        progress_bar = st.progress(0)
        
        # Step 1: Fetch content
        st.info(f"📥 Fetching {len(urls)} URLs...")
        reader = JinaReader()
        url_contents = {}
        
        for i, url in enumerate(urls):
            try:
                content = reader.fetch_url(url)
                if content.success:
                    url_contents[url] = content.content
                    st.success(f"✅ Fetched: {content.title}")
                else:
                    st.error(f"❌ Failed: {url} - {content.error}")
            except Exception as e:
                st.error(f"❌ Error fetching {url}: {str(e)}")
            
            progress_bar.progress((i + 1) / (len(urls) * 3))
        
        if not url_contents:
            st.error("❌ No URLs were successfully fetched!")
            return
        
        # Step 2: Extract product info
        st.info(f"🔍 Extracting product information...")
        extractor = ProductExtractor()
        products_dict = extractor.extract_multiple(url_contents)
        products = list(products_dict.values())
        
        progress_bar.progress(2 / 3)
        
        # Step 3: Store in vector database
        st.info(f"💾 Storing in vector database...")
        st.session_state.vector_store.add_products(products)
        st.session_state.products.extend(products)
        
        progress_bar.progress(1.0)
        
        st.success(f"🎉 Successfully added {len(products)} products!")
        time.sleep(0.5)
        progress_bar.empty()


def display_product(product, index):
    """Display a product card"""
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {index}. {product.title}")
            st.markdown(f"**🏢 Brand:** {product.brand or 'Unknown'}")
            
            if product.price:
                st.markdown(f"**💰 Price:** {product.currency}{product.price}")
            
            if product.rating:
                st.markdown(f"**⭐ Rating:** {product.rating}/5 ({product.review_count or 0} reviews)")
        
        with col2:
            if st.button(f"🗑️ Remove", key=f"remove_{index}"):
                st.session_state.vector_store.delete_by_url(product.url)
                st.session_state.products.pop(index - 1)
                st.rerun()
        
        # Expandable details
        with st.expander("📋 View Details"):
            if product.description:
                st.markdown("**Description:**")
                st.write(product.description[:300] + "..." if len(product.description) > 300 else product.description)
            
            if product.key_features:
                st.markdown("**✨ Key Features:**")
                for feature in product.key_features[:5]:
                    st.markdown(f"- {feature}")
            
            if product.specifications:
                st.markdown("**🔧 Specifications:**")
                spec_cols = st.columns(2)
                for i, (spec, value) in enumerate(list(product.specifications.items())[:6]):
                    with spec_cols[i % 2]:
                        st.markdown(f"**{spec}:** {value}")
        
        st.markdown("---")


def compare_products(query: str):
    """Run comparison query"""
    
    if not query:
        st.warning("⚠️ Please enter a question")
        return
    
    if len(st.session_state.products) == 0:
        st.warning("⚠️ Please add products first")
        return
    
    with st.spinner("🔬 Analyzing products..."):
        synthesizer = get_synthesizer()
        result = synthesizer.compare(query)
        
        # Add to history
        st.session_state.comparison_history.append({
            'query': query,
            'result': result,
            'timestamp': time.time()
        })
        
        return result


# ==================== MAIN APP ====================

# Header
st.markdown('<h1 class="main-header">🛍️ Product Comparison RAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Product Comparison using RAG & LLM</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📚 About")
    st.info("""
    This AI system helps you compare products intelligently by:
    - 🔗 Fetching data from any product URL
    - 🤖 Extracting key information using AI
    - 🔍 Semantic search across products
    - 💬 Natural language comparisons
    """)
    
    st.markdown("## 📊 Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><h3>🛍️</h3><p>Products</p><h2>' + 
                   str(len(st.session_state.products)) + '</h2></div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>💬</h3><p>Queries</p><h2>' + 
                   str(len(st.session_state.comparison_history)) + '</h2></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("## 🎯 Queries Instruction")
    st.markdown("""
    - Any Question which you think you want to ask about the product
    """)
    
    st.markdown("## ⚙️ Actions")
    if st.button("🗑️ Clear All Products"):
        st.session_state.vector_store.clear()
        st.session_state.products = []
        st.session_state.comparison_history = []
        st.success("✅ Cleared all data!")
        st.rerun()


# Main content
tab1, tab2, tab3 = st.tabs(["➕ Add Products", "💬 Compare Products", "📜 History"])

# ==================== TAB 1: Add Products ====================
with tab1:
    st.markdown("### 📥 Add Products to Compare")
    st.markdown("Paste product URLs from any website (Amazon, Flipkart, official sites, etc.)")
    
    # URL input
    url_input = st.text_area(
        "Enter product URLs (one per line)",
        height=150,
        # placeholder="https://www.apple.com/iphone-15/\nhttps://www.samsung.com/us/smartphones/galaxy-s24/\nhttps://www.oneplus.com/..."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Add Products", type="primary"):
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            if urls:
                add_products(urls)
            else:
                st.warning("⚠️ Please enter at least one URL")
    
    with col2:
        # Quick add example products
        if st.button("📱 Try Examples"):
            example_urls = [
                # "https://www.apple.com/iphone-15/",
                # "https://www.samsung.com/us/smartphones/galaxy-s24/"
            ]
            add_products(example_urls)
    
    st.markdown("---")
    
    # Display current products
    if st.session_state.products:
        st.markdown(f"### �� Current Products ({len(st.session_state.products)})")
        
        for i, product in enumerate(st.session_state.products, 1):
            display_product(product, i)
    else:
        st.info("�� Add some products above to get started!")


# ==================== TAB 2: Compare Products ====================
with tab2:
    st.markdown("### 💬 Ask Comparison Questions")
    
    if len(st.session_state.products) < 2:
        st.warning("⚠️ Add at least 2 products to start comparing")
    else:
        st.success(f"✅ Ready to compare {len(st.session_state.products)} products!")
        
        # Query input
        query = st.text_input(
            "What would you like to know?",
            # placeholder="Which phone has better camera quality?",
            key="comparison_query"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Compare", type="primary"):
                result = compare_products(query)
                if result:
                    st.markdown("---")
                    
                    # Display result
                    st.markdown('<div class="comparison-result">', unsafe_allow_html=True)
                    
                    # Query and products
                    st.markdown(f"### ❓ {result.query}")
                    st.markdown(f"**📱 Comparing:** {', '.join(result.products_compared)}")
                    
                    # Answer
                    st.markdown("### �� Answer")
                    st.markdown(result.answer)
                    
                    # Comparison table if available
                    if result.comparison_table:
                        st.markdown("### 📊 Quick Comparison")
                        st.json(result.comparison_table)
                    
                    # Citations
                    if result.citations:
                        with st.expander("🔗 View Sources"):
                            for i, citation in enumerate(result.citations, 1):
                                st.markdown(f"{i}. [{citation}]({citation})")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("🔄 Clear Query"):
                st.rerun()


# ==================== TAB 3: History ====================
with tab3:
    st.markdown("### 📜 Comparison History")
    
    if st.session_state.comparison_history:
        st.info(f"📊 Total comparisons: {len(st.session_state.comparison_history)}")
        
        # Reverse chronological order
        for i, item in enumerate(reversed(st.session_state.comparison_history), 1):
            result = item['result']
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['timestamp']))
            
            with st.expander(f"#{len(st.session_state.comparison_history) - i + 1} - {item['query']} ({timestamp})"):
                st.markdown(f"**📱 Products:** {', '.join(result.products_compared)}")
                st.markdown("**💬 Answer:**")
                st.markdown(result.answer)
                
                if result.comparison_table:
                    st.json(result.comparison_table)
        
        if st.button("🗑️ Clear History"):
            st.session_state.comparison_history = []
            st.rerun()
    else:
        st.info("No comparison history yet. Start comparing products in the Compare tab!")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p> Built using Streamlit, LangChain, ChromaDB & Groq(LLM)</p>
    <p>Made by Akash Mishra</p>
</div>
""", unsafe_allow_html=True)
