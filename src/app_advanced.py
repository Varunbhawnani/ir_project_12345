import streamlit as st
import sys
sys.path.append('.')
from search_engine_advanced import AdvancedSearchEngine, PersonalizationEngine
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import json
import hashlib

# Custom dictionary for book-related terms
BOOK_DOMAIN_DICTIONARY = {
    'rutherford': 'rutherford',
    'tolkien': 'tolkien',
    'rowling': 'rowling',
    'stephen': 'stephen',
    'king': 'king',
    'agatha': 'agatha',
    'christie': 'christie',
    'hemingway': 'hemingway',
    'fitzgerald': 'fitzgerald',
    'orwell': 'orwell',
    'asimov': 'asimov',
    'bradbury': 'bradbury',
    'fiction': 'fiction',
    'nonfiction': 'non-fiction',
    'thriller': 'thriller',
    'mystery': 'mystery',
    'romance': 'romance',
    'fantasy': 'fantasy',
    'scifi': 'sci-fi',
    'isbn': 'isbn',
}

# ============================================================
# CONFIGURATION - HARDCODED API KEY (DEVELOPMENT MODE)
# ============================================================
GROQ_API_KEY = ""  # REPLACE THIS WITH YOUR ACTUAL KEY

# Page config
st.set_page_config(
    page_title="E-Library Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed from "expanded" to "collapsed"
)

# [KEEP ALL THE EXISTING CSS - NO CHANGES]
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background with Image Overlay */
    .main {
        background: 
            linear-gradient(135deg, rgba(0, 0, 0, 0.65) 0%, rgba(10, 10, 10, 0.60) 100%),
            url('https://ik.imagekit.io/7xy7bon4y/Gemini_Generated_Image_uyz4umuyz4umuyz4.png') center/cover fixed;
    }
    
    .stApp {
        background: 
            linear-gradient(135deg, rgba(0, 0, 0, 0.65) 0%, rgba(10, 10, 10, 0.60) 100%),
            url('https://ik.imagekit.io/7xy7bon4y/Gemini_Generated_Image_uyz4umuyz4umuyz4.png') center/cover fixed;
    }
    
    /* Hero Header - Olive/Gold Theme */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, 
            rgba(20, 20, 20, 0.95) 0%, 
            rgba(30, 30, 30, 0.9) 50%,
            rgba(20, 20, 20, 0.95) 100%);
        padding: 4rem 2rem;
        margin-bottom: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(181, 168, 92, 0.2);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent, 
            #b5a85c, 
            #d4c57a,
            #b5a85c,
            transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #d4c57a 50%, #b5a85c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 1rem 0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    
    .main-header p {
        color: #b0b0b0;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0 auto;
        max-width: 700px;
        line-height: 1.6;
    }
    
    /* Remove old metric cards */
    .metrics-container {
        display: none;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 12px;
        animation: fadeIn 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(181, 168, 92, 0.2) 0%, rgba(212, 197, 122, 0.15) 100%);
        border-left: 3px solid #b5a85c;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #d4c57a;
    }
    
    .chat-message strong {
        color: #e5e7eb;
        font-weight: 600;
    }
    
    .source-book {
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 8px;
        border-left: 2px solid #b5a85c;
        font-size: 0.9rem;
        color: #d1d5db;
    }
    
    /* Result cards */
    /* Result cards - Target Streamlit containers */
[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.04) 0%, 
        rgba(255, 255, 255, 0.02) 100%);
    padding: 2rem;
    margin: 1.5rem 0;
    border-radius: 12px;
    border: 1px solid rgba(181, 168, 92, 0.15);
    transition: all 0.3s ease;
    position: relative;
    backdrop-filter: blur(10px);
    contain: layout;
    overflow: visible;
}

[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(180deg, #b5a85c, #d4c57a);
    border-radius: 12px 0 0 12px;
}

[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"]:hover {
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.06) 0%, 
        rgba(255, 255, 255, 0.04) 100%);
    border-color: rgba(181, 168, 92, 0.4);
    transform: translateX(4px);
    box-shadow: 0 8px 32px rgba(181, 168, 92, 0.2);
}

/* Keep old result-card for backwards compatibility */
.result-card {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.04) 0%, 
            rgba(255, 255, 255, 0.02) 100%);
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 12px;
        border: 1px solid rgba(181, 168, 92, 0.15);
        transition: all 0.3s ease;
        position: relative;
        backdrop-filter: blur(10px);
        /* ✅ Add containment */
        contain: layout;
        overflow: visible;
    }
    
     .element-container {
        width: 100% !important;
    }
    
    /* Fix stButton alignment inside result cards */
    .result-card .stButton {
        margin: 0;
    }
    
    /* Ensure metrics display properly */
    .result-card [data-testid="stMetricValue"] {
        font-size: 1rem;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, #b5a85c, #d4c57a);
        border-radius: 12px 0 0 12px;
    }
    
    .result-card:hover {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.06) 0%, 
            rgba(255, 255, 255, 0.04) 100%);
        border-color: rgba(181, 168, 92, 0.4);
        transform: translateX(4px);
        box-shadow: 0 8px 32px rgba(181, 168, 92, 0.2);
    }
    
    .result-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #f9fafb;
        margin-top: 0;
        font-weight: 600;
        font-size: 1.3rem;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
    
    /* Facet container */
    .facet-container {
        background: rgba(255, 255, 255, 0.04);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(181, 168, 92, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .facet-container h3 {
        font-weight: 600;
        font-size: 1rem;
        color: #f9fafb;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(181, 168, 92, 0.2);
    }
    
    /* Feature badges */
    .feature-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.3rem 0.3rem;
        background: linear-gradient(135deg, rgba(181, 168, 92, 0.25) 0%, rgba(212, 197, 122, 0.2) 100%);
        color: #d4c57a;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 1px solid rgba(181, 168, 92, 0.3);
    }
    
    /* Buttons - Olive/Gold Gradient */
    /* Buttons - Olive/Gold Gradient */
    .stButton > button {
        background: #5c5a3d;
        color: #d4c57a;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        border: 1px solid rgba(212, 197, 122, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: #4a4832;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(181, 168, 92, 0.4);
    }
    
    /* Input fields - Olive Theme */
    .stTextInput > div > div > input {
        background: rgba(181, 168, 92, 0.08);
        border: 1px solid rgba(181, 168, 92, 0.2);
        border-radius: 8px;
        color: #f9fafb !important;
        padding: 1rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #8a8a8a !important;
        opacity: 1 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #b5a85c;
        background: rgba(181, 168, 92, 0.12);
        box-shadow: 0 0 0 3px rgba(181, 168, 92, 0.15);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(181, 168, 92, 0.08);
        border: 1px solid rgba(181, 168, 92, 0.2);
        border-radius: 8px;
        color: #f9fafb;
        font-size: 0.95rem;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div:hover {
        border-color: #b5a85c;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 15, 0.95);
        border-right: 1px solid rgba(181, 168, 92, 0.15);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f9fafb;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid rgba(181, 168, 92, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #9ca3af;
        padding: 1rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #b5a85c;
    }
    
    .stTabs [aria-selected="true"] {
        color: #d4c57a;
        border-bottom: 2px solid #b5a85c;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #b5a85c 0%, #d4c57a 100%);
        border-radius: 4px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #d4c57a;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8a8a8a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(181, 168, 92, 0.08);
        border: 1px solid rgba(181, 168, 92, 0.2);
        border-radius: 8px;
        color: #f9fafb;
        font-weight: 500;
        padding: 1rem 1.25rem;
        font-size: 0.95rem;
        backdrop-filter: blur(10px);
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #b5a85c;
        background: linear-gradient(135deg, rgba(181, 168, 92, 0.15) 0%, rgba(212, 197, 122, 0.1) 100%);
    }
    
    /* Alert boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(181, 168, 92, 0.2);
        border-radius: 8px;
        color: #e5e7eb;
        font-size: 0.9rem;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(181, 168, 92, 0.15);
        border-radius: 8px;
        color: #e5e7eb;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #b5a85c, #d4c57a);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #a09450, #c4b56a);
    }
    
    /* Typography */
    p, span, div {
        color: #d1d5db;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .caption {
        color: #8a8a8a;
        font-size: 0.85rem;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #d1d5db;
    }
    
    .stCheckbox label {
        font-weight: 400;
        font-size: 0.9rem;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background: rgba(181, 168, 92, 0.08);
        border: 1px solid rgba(181, 168, 92, 0.2);
        border-radius: 8px;
        color: #f9fafb;
        font-weight: 500;
        text-align: center;
    }
    
    /* Section divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(181, 168, 92, 0.4), transparent);
        margin: 2rem 0;
    }
    
    /* Headings */
    h1, h2, h3, h4 {
        color: #f9fafb;
        font-weight: 600;
    }
    
    h2 {
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    h3 {
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    /* Layout */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Alert colors - Olive themed */
    .stSuccess {
        background: linear-gradient(135deg, rgba(181, 168, 92, 0.2) 0%, rgba(212, 197, 122, 0.15) 100%);
        border: 1px solid rgba(181, 168, 92, 0.4);
        color: #d4c57a;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(181, 168, 92, 0.2) 0%, rgba(212, 197, 122, 0.15) 100%);
        border: 1px solid rgba(181, 168, 92, 0.5);
        color: #d4c57a;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(181, 168, 92, 0.15) 0%, rgba(212, 197, 122, 0.1) 100%);
        border: 1px solid rgba(181, 168, 92, 0.3);
        color: #b5a85c;
    }
    
    /* Analytics graph containers */
/* Analytics graph containers */
.stPlotlyChart {
    background: linear-gradient(135deg, 
        rgba(255, 255, 255, 0.04) 0%, 
        rgba(255, 255, 255, 0.02) 100%);
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid rgba(181, 168, 92, 0.15);
    backdrop-filter: blur(10px);
    margin: 1.5rem 0;
}

/* Ensure plotly charts don't overflow */
.stPlotlyChart > div {
    overflow: visible !important;
}

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
if 'ab_test_variant' not in st.session_state:
    st.session_state.ab_test_variant = 'A' if hash(st.session_state.user_id) % 2 == 0 else 'B'
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []
if 'compared_results' not in st.session_state:
    st.session_state.compared_results = {}
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'search'  # 'search' or 'chat'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_search_query' not in st.session_state:
    st.session_state.last_search_query = ""
if 'vsm_weight' not in st.session_state:
    st.session_state.vsm_weight = 0.4  # Default VSM weight
if 'semantic_weight' not in st.session_state:
    st.session_state.semantic_weight = 0.6  # Default Semantic weight
if 'use_query_expansion_global' not in st.session_state:
    st.session_state.use_query_expansion_global = True

# Load search engine
# Load search engine
@st.cache_resource
def load_search_engine():
    engine = AdvancedSearchEngine()
    try:
        engine.load_model('../data/advanced_search_engine.pkl')
    except Exception as e:
        print(f"Could not load model, building from scratch: {e}")
        engine.load_data('../data/books.csv', num_docs=5000)
        engine.build_vsm_index()
        engine.build_semantic_index()
        engine.save_model()
    
    # IMPORTANT: Create a NEW session-based personalization engine (not saved)
    if hasattr(engine, 'semantic_embeddings') and engine.semantic_embeddings is not None:
        engine.personalization = PersonalizationEngine(session_based=True)
        # Restore item embeddings from the loaded model
        for idx, embedding in enumerate(engine.semantic_embeddings):
            engine.personalization.item_embeddings[idx] = embedding
    
    return engine

# Load RAG system
# Load RAG system
@st.cache_resource
def load_rag_system():
    try:
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
            return None
        
        # Try to load multimodal RAG first
        try:
            from rag_llm_multimodal import MultimodalRAGWithLLM
            rag = MultimodalRAGWithLLM(api_key=GROQ_API_KEY)
            print("✅ Using Multimodal RAG (text + visual)")
            return rag
        except:
            # Fallback to text-only RAG
            from rag_llm import RAGWithLLM
            rag = RAGWithLLM(api_key=GROQ_API_KEY)
            print("✅ Using Text-only RAG")
            return rag
    except Exception as e:
        print(f"Error loading RAG: {e}")
        return None

# Load engine
try:
    with st.spinner("Loading search engine..."):
        engine = load_search_engine()
    engine_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    engine_loaded = False

# Load RAG system
rag_system = load_rag_system()
rag_available = rag_system is not None

# Load multimodal search engine
@st.cache_resource
def load_multimodal_engine():
    """Load Kaggle book cover search engine"""
    try:
        import sys
        sys.path.append('.')
        from multimodal_search_kaggle import KaggleBookCoverSearch
        
        engine = KaggleBookCoverSearch()
        engine.load_model('../data/kaggle_multimodal_search.pkl')
        return engine
    except Exception as e:
        print(f"Multimodal search not available: {e}")
        return None

multimodal_engine = load_multimodal_engine()
multimodal_available = multimodal_engine is not None

# Hero Header with Mode Toggle
# Hero Header (Centered)
# ============================================================
# CLEAN HERO SECTION
# ============================================================

# Centered Header
st.markdown("""
<div class='main-header'>
    <h1>E-Library Search</h1>
    <p>Discover books with intelligent search powered by hybrid algorithms, semantic understanding, and personalized recommendations</p>
</div>
""", unsafe_allow_html=True)

# Mode Selector - Clean and centered
st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([2, 3, 2])
with col_center:
    mode_options = ["🔍 Search Mode"]
    if rag_available:
        mode_options.append("💬 Chat Mode")
    if multimodal_available:
        mode_options.append("🎨 Visual Search")
    
    selected_mode = st.selectbox(
        "Choose Mode",
        mode_options,
        index=0,
        help="Select your preferred search interface"
    )
    
    if '🔍' in selected_mode:
        st.session_state.app_mode = 'search'
    elif '💬' in selected_mode:
        st.session_state.app_mode = 'chat'
    elif '🎨' in selected_mode:
        st.session_state.app_mode = 'visual'

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

if engine_loaded:
    # Compact stats bar
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # ============================================================
    # CHAT MODE
    # ============================================================
    if st.session_state.app_mode == 'chat' and rag_available:
        st.markdown("---")
        
        # Sidebar for chat mode
        with st.sidebar:
            st.header("💬 Chat Assistant")
            
            st.info("Ask me anything about books! I'll search the library and provide personalized recommendations.")
            
            # Show current search settings being used
            with st.expander("🔧 Current Search Settings"):
                st.metric("VSM Weight", f"{st.session_state.vsm_weight:.1f}")
                st.metric("Semantic Weight", f"{st.session_state.semantic_weight:.1f}")
                st.metric("Query Expansion", "✓ Enabled" if st.session_state.use_query_expansion_global else "✗ Disabled")
                st.caption("💡 These settings are inherited from Search Mode")
            
            if st.button("🔄 Clear Conversation", width="stretch"):
                st.session_state.chat_history = []
                st.rerun()
            
            st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
            
            if st.session_state.last_search_query:
                st.markdown("---")
                st.markdown("**💡 Quick Actions**")
                if st.button(f"Ask about: '{st.session_state.last_search_query[:30]}...'",
                           width="stretch"):
                    st.session_state.chat_input_value = st.session_state.last_search_query
                    st.rerun()
        
        # Main chat interface
        st.markdown("## 💬 Chat with AI Librarian")
        
        # Display chat history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class='chat-message user-message'>
                        <strong>🧑 You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-message assistant-message'>
                        <strong>🤖 AI Librarian:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show source books if available
                    if 'books' in message and message['books']:
                        with st.expander(f"📚 Source Books ({len(message['books'])}) - Dataset: {message['books'][0].get('source', 'unknown')}"):
                            for book in message['books']:
                    # Get score (different field names for different datasets)
                                score = book.get('hybrid_score', book.get('score', 0))
                                dataset = book.get('source', 'unknown')
            
                                st.markdown(f"""
                                <div class='source-book'>
                                    <strong>{book['document']['title']}</strong><br>
                                    <em>by {book['document']['author']}</em><br>
                                    Relevance Score: {score:.3f}
                                    {f"<br>Dataset: {dataset}" if dataset != 'unknown' else ''}
                                    {f"<br>Visual Score: {book.get('visual_score', 0):.3f}" if 'visual_score' in book else ''}
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("👋 Hello! I'm your AI Librarian. Ask me about books, and I'll help you find exactly what you're looking for!")
        
        # Chat input
        st.markdown("---")
        user_input = st.text_input(
            "Your message",
            placeholder="Example: I want mystery novels with strong female detectives",
            key="chat_input",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            send_button = st.button("💬 Send", type="primary", width="stretch")
        with col2:
            if st.button("🔄 New Topic", width="stretch"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process chat
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get RAG response
            # Get RAG response with custom weights
            with st.spinner("🤖 AI Librarian is thinking..."):
                try:
                    # Pass the session state weights to the RAG system
                    response, books = rag_system.chat(
                        user_input,
                        vsm_weight=st.session_state.vsm_weight,
                        semantic_weight=st.session_state.semantic_weight,
                        use_expansion=st.session_state.use_query_expansion_global
                    )
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'books': books
                    })
                    
                    # Store query for potential use in search mode
                    st.session_state.last_search_query = user_input
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.",
                        'books': []
                    })
            
            st.rerun()
            
            # ============================================================
    # VISUAL SEARCH MODE (NEW!)
    # ============================================================
    elif st.session_state.app_mode == 'visual' and multimodal_available:
        st.markdown("---")
        st.markdown("## 🎨 Visual Book Cover Search")
        st.info("🔍 Search for books by describing what the cover looks like! Try: 'blue book cover', 'dark mysterious design', 'book with a face'")
        
        # Sidebar for visual search
        with st.sidebar:
            st.header("🎨 Visual Search")
            st.markdown("**Search by:**")
            st.caption("• Colors: 'blue cover', 'red book'")
            st.caption("• Style: 'minimalist', 'vintage'")
            st.caption("• Content: 'book with face', 'landscape'")
            st.caption("• Mood: 'dark mysterious', 'cheerful'")
        
        # Search input
        visual_query = st.text_input(
            "Describe the book cover you're looking for:",
            placeholder="blue colored book cover, dark mysterious design, vintage style...",
            key="visual_search_input"
        )
        
        # Show spell-checked suggestions if query exists
        # Show spell-checked suggestions if query exists
        if visual_query:
            from spellchecker import SpellChecker
            import re
            
            spell = SpellChecker()
            
            # EXPANDED DICTIONARY: Visual terms + Book genres/content terms
            visual_and_semantic_dict = {
                # Color words
                'blue', 'green', 'red', 'yellow', 'orange', 'purple', 'pink',
                'brown', 'black', 'white', 'gray', 'grey', 'cyan', 'navy',
                'turquoise', 'crimson', 'scarlet', 'emerald', 'golden', 'silver',
                'dark', 'light', 'bright', 'pale', 'deep', 'vivid', 'neon',
                
                # Visual style terms
                'minimalist', 'vintage', 'modern', 'abstract', 'realistic',
                'colorful', 'monochrome', 'gradient', 'illustration', 'photographic',
                'artistic', 'simple', 'complex', 'elegant', 'bold', 'subtle',
                
                # Book/cover terms
                'cover', 'book', 'design', 'artwork', 'image', 'illustration',
                'picture', 'photo', 'graphic', 'styled', 'designed',
                
                # SEMANTIC TERMS (same as Search Mode)
                'mystery', 'thriller', 'horror', 'romance', 'fiction', 'fantasy',
                'science', 'detective', 'crime', 'love', 'magic', 'adventure',
                'historical', 'biography', 'comedy', 'drama', 'action', 'suspense',
                'supernatural', 'dystopian', 'contemporary', 'classic', 'young',
                'adult', 'children', 'novel', 'story', 'tales', 'saga',
                
                # Authors (common misspellings)
                'tolkien', 'rowling', 'stephen', 'king', 'agatha', 'christie',
                'hemingway', 'fitzgerald', 'orwell', 'asimov', 'bradbury',
                'rutherford'
            }
            spell.word_frequency.load_words(visual_and_semantic_dict)
            
            # INTELLIGENT SEGMENTATION: Handle both visual and semantic queries
            def segment_visual_query(text):
                patterns = [
                    # Visual patterns
                    (r'(blue|green|red|yellow|orange|purple|pink)(book|cover)', r'\1 \2'),
                    (r'(dark|light|bright)(blue|green|red|yellow|purple|orange)', r'\1 \2'),
                    (r'(book)(cover)', r'book cover'),
                    (r'(cover)(design)', r'cover design'),
                    
                    # Semantic patterns (from Search Mode)
                    (r'(books?by)(\w+)', r'books by \2'),
                    (r'(authorname)(\w+)', r'author name \2'),
                    (r'(isbn)(\d+)', r'isbn \2'),
                    (r'(\w+)(fiction|mystery|thriller|romance|horror|fantasy)', r'\1 \2'),
                    (r'(science)(fiction)', r'science fiction'),
                    (r'(best)(seller)', r'best seller'),
                ]
                
                result = text.lower()
                for pattern, replacement in patterns:
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                # If still no spaces and length > 10, try to split
                if ' ' not in result and len(result) > 10:
                    for i in range(3, min(len(result), 8)):
                        prefix = result[:i]
                        suffix = result[i:]
                        
                        if prefix in spell or prefix in visual_and_semantic_dict:
                            if len(suffix) >= 3:
                                return f"{prefix} {suffix}"
                
                return result
            
            segmented_query = segment_visual_query(visual_query)
            
            # Spell check
            words = segmented_query.lower().split()
            ignore_words = visual_and_semantic_dict | {'isbn', 'books', 'book', 'by', 'the', 'a', 'an', 'of', 'with'}
            misspelled = [w for w in spell.unknown(words) if w not in ignore_words and not w.isdigit()]
            
            corrected_queries = set()
            
            # If segmentation changed the query, that's a suggestion
            if segmented_query != visual_query.lower() and ' ' in segmented_query:
                corrected_queries.add(segmented_query)
            
            # Spell correction on individual words
            if misspelled:
                for misspelled_word in list(misspelled)[:2]:
                    candidates = spell.candidates(misspelled_word)
                    
                    if candidates:
                        for candidate in list(candidates)[:3]:
                            corrected_words = []
                            for word in words:
                                if word == misspelled_word:
                                    corrected_words.append(candidate)
                                else:
                                    if word in misspelled:
                                        correction = spell.correction(word)
                                        corrected_words.append(correction if correction else word)
                                    else:
                                        corrected_words.append(word)
                            
                            corrected_query = ' '.join(corrected_words)
                            if corrected_query.lower() != visual_query.lower():
                                corrected_queries.add(corrected_query)
                            
                            if len(corrected_queries) >= 3:
                                break
                    
                    if len(corrected_queries) >= 3:
                        break
            
            # Common corrections for BOTH visual and semantic queries
            common_phrases = {
                # Visual phrases
                'bluecover': 'blue cover',
                'redbook': 'red book',
                'darkblue': 'dark blue',
                'lightgreen': 'light green',
                'bookcover': 'book cover',
                'coverdesign': 'cover design',
                
                # Semantic phrases (from Search Mode)
                'booksbyrutherford': 'books by rutherford',
                'booksby': 'books by',
                'authorname': 'author name',
                'sciencefiction': 'science fiction',
                'mysterythriller': 'mystery thriller',
                'bestbook': 'best book',
                'horrorstory': 'horror story',
            }
            
            query_normalized = visual_query.lower().replace(' ', '')
            for pattern, suggestion in common_phrases.items():
                if pattern in query_normalized:
                    corrected_queries.add(suggestion)
            
            # Segment query intelligently
            def segment_visual_query(text):
                patterns = [
                    (r'(blue|green|red|yellow)(book|cover)', r'\1 \2'),
                    (r'(dark|light|bright)(blue|green|red)', r'\1 \2'),
                    (r'(book)(cover)', r'book cover'),
                ]
                
                result = text.lower()
                for pattern, replacement in patterns:
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
                return result
            
            segmented_query = segment_visual_query(visual_query)
            
            # Spell check
            words = segmented_query.lower().split()
            ignore_words = visual_and_semantic_dict | {'isbn', 'books', 'book', 'by', 'cover', 'the', 'a', 'an'}
            misspelled = [w for w in spell.unknown(words) if w not in ignore_words and not w.isdigit()]
            
            corrected_queries = set()
            
            # If segmentation changed the query, that's a suggestion
            if segmented_query != visual_query.lower() and ' ' in segmented_query:
                corrected_queries.add(segmented_query)
            
            # Spell correction
            if misspelled:
                for misspelled_word in list(misspelled)[:2]:
                    candidates = spell.candidates(misspelled_word)
                    
                    if candidates:
                        for candidate in list(candidates)[:2]:
                            corrected_words = []
                            for word in words:
                                if word == misspelled_word:
                                    corrected_words.append(candidate)
                                else:
                                    if word in misspelled:
                                        correction = spell.correction(word)
                                        corrected_words.append(correction if correction else word)
                                    else:
                                        corrected_words.append(word)
                            
                            corrected_query = ' '.join(corrected_words)
                            if corrected_query.lower() != visual_query.lower():
                                corrected_queries.add(corrected_query)
                            
                            if len(corrected_queries) >= 3:
                                break
                    
                    if len(corrected_queries) >= 3:
                        break
            
            # Common visual search phrase corrections
            common_visual_phrases = {
                'bluecover': 'blue cover',
                'redbook': 'red book',
                'darkblue': 'dark blue',
                'lightgreen': 'light green',
                'bookcover': 'book cover',
                'coverdesign': 'cover design',
            }
            
            query_normalized = visual_query.lower().replace(' ', '')
            for pattern, suggestion in common_visual_phrases.items():
                if pattern in query_normalized:
                    corrected_queries.add(suggestion)
            
            # Show suggestions
            if corrected_queries:
                corrected_queries = {s for s in corrected_queries if s.lower() != visual_query.lower()}
                
                if corrected_queries:
                    st.markdown("**💡 Did you mean?**")
                    sug_cols = st.columns(min(len(corrected_queries), 3))
                    for i, suggestion in enumerate(list(corrected_queries)[:3]):
                        with sug_cols[i]:
                            if st.button(suggestion, key=f"visual_spell_sug_{i}", use_container_width=True):
                                st.session_state.visual_rerun_query = suggestion
                                st.session_state.visual_auto_search = True
                                st.rerun()
        
        col1, col2 = st.columns([1, 4])
        with col1:
            visual_search_button = st.button("🔍 Search Covers", type="primary", use_container_width=True)
        with col2:
            num_results = st.number_input("Results", min_value=5, max_value=20, value=10, label_visibility="visible")
        
        # Example queries
        st.markdown("### 💡 Try These Examples:")
        example_cols = st.columns(4)
        visual_examples = [
            "blue book cover",
            "dark mysterious cover",
            "minimalist design",
            "book with a face"
        ]
        
        for i, ex in enumerate(visual_examples):
            with example_cols[i]:
                if st.button(ex, key=f"visual_ex_{i}", use_container_width=True):
                    st.session_state.visual_rerun_query = ex
                    st.session_state.visual_auto_search = True
                    st.rerun()
        
        # Perform visual search
        # Check if auto-search was triggered
        visual_auto_search = st.session_state.get('visual_auto_search', False)
        if visual_auto_search:
            visual_search_button = True
            del st.session_state.visual_auto_search
        
        # If auto-search is triggered, use the rerun_query if it exists
        if visual_auto_search and hasattr(st.session_state, 'visual_rerun_query'):
            visual_query = st.session_state.visual_rerun_query
            del st.session_state.visual_rerun_query
        
        # Perform visual search
        if visual_search_button and visual_query:
            with st.spinner("🔍 Searching through 30,000+ book covers..."):
                try:
                    results = multimodal_engine.search_by_visual_description(visual_query, top_k=num_results)
                    
                    # Validate results
                    if not results:
                        st.warning("No results found. Try a different description.")
                        results = []
                    
                    st.markdown(f"### 📚 Found {len(results)} Visually Matching Covers")
                    st.caption(f"Query: '{visual_query}'")
                    
                    # Display results in grid (3 columns)
                    for i in range(0, len(results), 3):
                        cols = st.columns(3)
                        
                        for j, result in enumerate(results[i:i+3]):
                            with cols[j]:
                                st.markdown(f"**{result['rank']}. {result['title'][:40]}...**")
                                
                                # Show book cover image
                                try:
                                    from PIL import Image
                                    img = Image.open(result['image_path'])
                                    st.image(img, width="stretch")
                                except Exception as e:
                                    st.error("📷 Image not available")
                                
                                # Book details
                                st.caption(f"✍️ by {result['author'][:30]}")
                                st.caption(f"📂 {result['category']}")
                                
                                # UPDATED: Show breakdown of scores
                                # UPDATED: Show breakdown of scores
                                if 'color_score' in result:
                                    # Color-based search
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        similarity = result.get('similarity_score', 0)
                                        if isinstance(similarity, (int, float)) and not np.isnan(similarity):
                                            st.metric("Overall", f"{similarity:.1%}")
                                        else:
                                            st.metric("Overall", "N/A")
                                    with col2:
                                        color_score = result.get('color_score', 0)
                                        if isinstance(color_score, (int, float)) and not np.isnan(color_score):
                                            st.metric("Color Match", f"{color_score:.1%}")
                                        else:
                                            st.metric("Color Match", "N/A")
                                                                    
                                    # Show dominant color
                                    # Show dominant color
                                    if 'dominant_rgb' in result:
                                        rgb = result['dominant_rgb']
                                        # Handle both list and numpy array cases
                                        try:
                                            if isinstance(rgb, (list, tuple, np.ndarray)) and len(rgb) >= 3:
                                                r_val = int(rgb[0]) if not np.isnan(rgb[0]) else 128
                                                g_val = int(rgb[1]) if not np.isnan(rgb[1]) else 128
                                                b_val = int(rgb[2]) if not np.isnan(rgb[2]) else 128
                                                
                                                st.markdown(
                                                    f"<div style='background-color: rgb({r_val},{g_val},{b_val}); "
                                                    f"height: 20px; border-radius: 5px; border: 1px solid #ccc;'></div>",
                                                    unsafe_allow_html=True
                                                )
                                                st.caption(f"RGB: ({r_val}, {g_val}, {b_val})")
                                        except (TypeError, IndexError, KeyError):
                                            pass  # Skip if color data is malformed
                                else:
                                    # Semantic search only
                                    st.metric("Semantic Match", f"{result['similarity_score']:.1%}")
                                
                                # Additional info in expander
                               # Additional info in expander
                                # Additional info in expander
                                with st.expander("📖 More Info"):
                                    try:
                                        rating = result.get('rating')
                                        if rating and rating != 'N/A' and (not isinstance(rating, float) or not pd.isna(rating)):
                                            st.write(f"⭐ Rating: {rating}")
                                    except:
                                        pass
                                    
                                    try:
                                        price = result.get('price')
                                        if price and price != 'N/A':
                                            st.write(f"💰 Price: {price}")
                                    except:
                                        pass
                                    
                                    try:
                                        isbn = result.get('isbn')
                                        if isbn and isbn != 'N/A':
                                            st.write(f"📚 ISBN: {isbn}")
                                    except:
                                        pass
                                    
                                    if 'clip_score' in result:
                                        st.write(f"🔍 CLIP Score: {result['clip_score']:.3f}")
                                        if 'color_score' in result:
                                            st.write(f"🎨 Color Score: {result['color_score']:.3f}")
                                
                                st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Search error: {e}")
        
        elif not visual_query and visual_search_button:
            st.warning("⚠️ Please enter a description of the book cover you're looking for!")
    
    # ============================================================
    # SEARCH MODE (EXISTING CODE)
    # ============================================================
    elif st.session_state.app_mode == 'search':
        # Search Controls in Expander (cleaner than sidebar)
        with st.expander("⚙️ Advanced Search Controls", expanded=False):
            ctrl_col1, ctrl_col2 = st.columns(2)
            
            with ctrl_col1:
                st.subheader("🔍 Search Algorithm")
                search_mode = st.selectbox(
                    "Algorithm",
                    ["Hybrid + Query Expansion",
                     "Hybrid Search",
                     "VSM + BM25",
                     "Semantic Search",
                     "Personalized Search"],
                    label_visibility="collapsed"
                )
                
                if "Hybrid" in search_mode:
                    vsm_weight = st.slider("VSM Weight", 0.0, 1.0,
                                          st.session_state.vsm_weight, 0.1,
                                          key="vsm_slider")
                    semantic_weight = 1.0 - vsm_weight
                    st.session_state.vsm_weight = vsm_weight
                    st.session_state.semantic_weight = semantic_weight
                    st.caption(f"Semantic Weight: {semantic_weight:.1f}")
                else:
                    vsm_weight = st.session_state.vsm_weight
                    semantic_weight = st.session_state.semantic_weight
                
                use_personalization = st.checkbox("Enable Personalization", value=True)
                use_query_expansion = st.checkbox("Enable Query Expansion",
                                                 value="Expansion" in search_mode)
                st.session_state.use_query_expansion_global = use_query_expansion
                
                if use_query_expansion:
                    expansion_terms = st.slider("Expansion Terms", 3, 10, 5)
            
            with ctrl_col2:
                st.subheader("📊 Quick Stats")
                
                if st.session_state.search_history:
                    st.metric("Total Searches", len(st.session_state.search_history))
                    recent = st.session_state.search_history[-1]
                    st.metric("Last Search", f"{recent.get('search_time', 0):.3f}s")
                
                st.metric("Bookmarks", len(st.session_state.bookmarks))
                
                st.markdown("---")
                
                if st.button("📊 View Analytics", width="stretch"):
                    st.session_state.show_analytics = True
                
                enable_ab_test = st.checkbox("A/B Testing")
                if enable_ab_test:
                    st.info(f"Variant: {st.session_state.ab_test_variant}")
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Main search interface
        # Prominent Search Bar
        col_search, col_num = st.columns([8, 1])
        
        with col_search:
            # Check if we have a corrected query to use
            if 'rerun_query' in st.session_state:
                default_query = st.session_state.rerun_query
                # Clear it from session state
                del st.session_state.rerun_query
                # Force update the text input by changing its key
                input_key = f"main_search_{hash(default_query)}"
            else:
                default_query = ""
                input_key = "main_search"
            
            query = st.text_input(
                "Search",
                placeholder="Search for mystery detective, science fiction, romantic stories...",
                value=default_query,
                key=input_key,
                label_visibility="collapsed"
            )
            
            # Store query for chat mode
            if query:
                st.session_state.last_search_query = query
        
        with col_num:
            top_k = st.number_input("Results", min_value=5, max_value=50, value=10, label_visibility="visible")
        
        # Show spell-checked suggestions if query exists
        # Show spell-checked suggestions if query exists
        if query:
            from spellchecker import SpellChecker
            import re
            
            spell = SpellChecker()
            spell.word_frequency.load_words(BOOK_DOMAIN_DICTIONARY.keys())
            
            # ===== STEP 1: Intelligent Query Segmentation =====
            def segment_query(text):
                """Intelligently segment concatenated words"""
                # Common search patterns
                patterns = [
                    (r'(books?by)(\w+)', r'books by \2'),  # booksby -> books by
                    (r'(authorname)(\w+)', r'author name \2'),  # authorname -> author name
                    (r'(isbn)(\d+)', r'isbn \2'),  # isbn123 -> isbn 123
                    (r'(\w+)(fiction|mystery|thriller|romance|horror|fantasy)', r'\1 \2'),  # sciencefiction
                    (r'(science)(fiction)', r'science fiction'),
                    (r'(best)(seller)', r'best seller'),
                    (r'(new)(york)', r'new york'),
                ]
                
                original = text.lower()
                segmented = original
                
                for pattern, replacement in patterns:
                    segmented = re.sub(pattern, replacement, segmented, flags=re.IGNORECASE)
                
                # If still no spaces and length > 10, try to split at common word boundaries
                if ' ' not in segmented and len(segmented) > 10:
                    # Try splitting into known words
                    for i in range(3, min(len(segmented), 8)):
                        prefix = segmented[:i]
                        suffix = segmented[i:]
                        
                        # Check if prefix is a valid word
                        if prefix in spell or prefix in ['books', 'book', 'author', 'isbn', 'by', 'the', 'of']:
                            if len(suffix) >= 3:
                                return f"{prefix} {suffix}"
                
                return segmented
            
            # Segment the query first
            segmented_query = segment_query(query)
            
            # ===== STEP 2: Spell Checking on Segmented Query =====
            words = segmented_query.lower().split()
            
            # Find misspelled words (but ignore common search terms)
            ignore_words = {'isbn', 'books', 'book', 'by', 'author', 'fiction', 'sci', 'fi'}
            misspelled = [w for w in spell.unknown(words) if w not in ignore_words and not w.isdigit()]
            
            corrected_queries = set()
            
            # If segmentation changed the query, that's a suggestion
            if segmented_query != query.lower() and ' ' in segmented_query:
                corrected_queries.add(segmented_query)
            
            # Spell correction on individual words
            if misspelled:
                # Generate multiple correction possibilities
                for misspelled_word in list(misspelled)[:2]:  # Handle up to 2 misspelled words
                    # Get top candidates for this misspelled word
                    candidates = spell.candidates(misspelled_word)
                    
                    if candidates:
                        # Take top 2-3 candidates for this word
                        for candidate in list(candidates)[:3]:
                            corrected_words = []
                            for word in words:
                                if word == misspelled_word:
                                    corrected_words.append(candidate)
                                else:
                                    # Check if this word is also misspelled
                                    if word in misspelled:
                                        correction = spell.correction(word)
                                        corrected_words.append(correction if correction else word)
                                    else:
                                        corrected_words.append(word)
                            
                            corrected_query = ' '.join(corrected_words)
                            if corrected_query.lower() != query.lower():
                                corrected_queries.add(corrected_query)
                            
                            if len(corrected_queries) >= 3:
                                break
                    
                    if len(corrected_queries) >= 3:
                        break
            
            # ===== STEP 3: Context-Aware Suggestions =====
            # Check against common book-related phrases
            common_phrases = {
                'booksbyrutherford': 'books by rutherford',
                'booksby': 'books by',
                'authorname': 'author name',
                'sciencefiction': 'science fiction',
                'mysterythriller': 'mystery thriller',
                'bestbook': 'best book',
                'newrelease': 'new release',
                'toprated': 'top rated',
            }
            
            query_normalized = query.lower().replace(' ', '')
            for pattern, suggestion in common_phrases.items():
                if pattern in query_normalized:
                    corrected_queries.add(suggestion)
            
            # Show suggestions
            if corrected_queries:
                # Remove exact duplicates and the original query
                corrected_queries = {s for s in corrected_queries if s.lower() != query.lower()}
                
                if corrected_queries:
                    st.markdown("**💡 Did you mean?**")
                    sug_cols = st.columns(min(len(corrected_queries), 3))
                    for i, suggestion in enumerate(list(corrected_queries)[:3]):
                        with sug_cols[i]:
                            if st.button(suggestion, key=f"spell_sug_{i}", width="stretch"):
                                # Auto-fill and trigger search
                                st.session_state.rerun_query = suggestion
                                st.session_state.auto_search = True
                                st.rerun()
        
        # Search buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 1])
        with btn_col1:
            search_clicked = st.button("🔍 Search", type="primary", width="stretch")
        with btn_col2:
            lucky_clicked = st.button("🎲 I'm Feeling Lucky", width="stretch")
        with btn_col3:
            pass
        with btn_col4:
            if st.button("🔄 Clear", width="stretch"):
                st.rerun()
        
        # Check if auto-search was triggered
        auto_search = st.session_state.get('auto_search', False)
        if auto_search:
            search_clicked = True
            del st.session_state.auto_search
        
        # Popular searches
        st.markdown("### 💡 Popular Searches")
        example_cols = st.columns(6)
        examples = [
            "mystery thriller", "science fiction", "romance novel",
            "horror stories", "fantasy magic", "business finance"
        ]
        
        for i, ex in enumerate(examples):
            with example_cols[i]:
                if st.button(ex, key=f"ex_{i}", width="stretch"):
                    query = ex
                    search_clicked = True
        
        # Perform search
        # Perform search
        if (search_clicked or query or lucky_clicked or auto_search) and (query or lucky_clicked):
            if lucky_clicked:
                import random
                query = random.choice(examples)
            
            # If auto_search is triggered, use the rerun_query if it exists
            if auto_search and hasattr(st.session_state, 'rerun_query'):
                query = st.session_state.rerun_query
            
            intent = engine.intent_classifier.classify(query)
            
            # Show search strategy based on intent
            if intent == 'navigational':
                st.info("🎯 **Navigational Search**: Looking for exact match (ISBN/specific book)")
            elif intent == 'author_search':
                st.info("🎯 **Author Search**: Searching by author name")
            
            st.markdown("---")
            analysis_cols = st.columns(4)
            with analysis_cols[0]:
                st.info(f"🎯 **Intent:** {intent.title()}")
            with analysis_cols[1]:
                st.info(f"🔍 **Mode:** {search_mode.split('+')[0].strip()}")
            
            start_time = time.time()
            
            with st.spinner("🔍 Searching through library..."):
                try:
                    if "VSM" in search_mode:
                        results = engine.vsm_search(query, top_k)
                    elif "Semantic" in search_mode:
                        results = engine.semantic_search(query, top_k)
                    elif "Personalized" in search_mode:
                        results = engine.hybrid_search(
                            query, top_k,
                            use_expansion=use_query_expansion,
                            personalized_for_user=st.session_state.user_id
                        )
                    else:
                        if enable_ab_test and st.session_state.ab_test_variant == 'B':
                            results = engine.hybrid_search(
                                query, top_k,
                                vsm_weight=0.3, semantic_weight=0.7,
                                use_expansion="Expansion" in search_mode,
                                personalized_for_user=st.session_state.user_id
                            )
                        else:
                            results = engine.hybrid_search(
                                query, top_k,
                                use_expansion="Expansion" in search_mode,
                                personalized_for_user=st.session_state.user_id if use_personalization else None
                            )
                    
                    search_time = time.time() - start_time
                    
                    st.session_state.search_history.append({
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'results_count': len(results),
                        'search_time': search_time,
                        'intent': intent
                    })
                    
                    if results and results[0].get('expanded_terms'):
                        with analysis_cols[2]:
                            st.success(f"📝 **Expanded:** +{len(results[0]['expanded_terms'])} terms")
                        with analysis_cols[3]:
                            st.info(f"⚡ **Time:** {search_time:.3f}s")
                    else:
                        with analysis_cols[3]:
                            st.info(f"⚡ **Time:** {search_time:.3f}s")
                            
                    if use_personalization and results:
                        for result in results[:3]:
                            engine.personalization.update_user_profile(
                                st.session_state.user_id,
                                result['document']['id'],
                                'view', 0.1
                            )
                            
                except Exception as e:
                    st.error(f"❌ Search error: {e}")
                    results = []
            
            if results:
                facets = engine.get_facets(results)
                
                st.markdown(f"### 📚 Found {len(results)} Results")
                
                tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Analytics", "🔬 Compare Methods"])
                
                with tab1:
                    for r in results:
                        # Create a styled container using Streamlit's native container
                        with st.container():
                            # Title and bookmark row
                            title_col, bookmark_col = st.columns([5, 1])
                            with title_col:
                                st.markdown(f"### {r['rank']}. {r['document']['title']}")
                            with bookmark_col:
                                if st.button("📌", key=f"bookmark_{r['document']['id']}"):
                                    st.session_state.bookmarks.append({
                                        'title': r['document']['title'],
                                        'id': r['document']['id']
                                    })
                            
                            # Metadata row
                            meta_cols = st.columns(4)
                            with meta_cols[0]:
                                st.caption(f"**Author:** {r['document']['author']}")
                            with meta_cols[1]:
                                if 'genre' in r['document']:
                                    st.caption(f"**Genre:** {r['document'].get('genre', 'General')}")
                            with meta_cols[2]:
                                if 'year' in r['document']:
                                    st.caption(f"**Year:** {r['document'].get('year', 'N/A')}")
                            with meta_cols[3]:
                                if 'rating' in r['document']:
                                    rating = r['document'].get('rating', 0)
                                    stars = "⭐" * int(rating)
                                    st.caption(f"**Rating:** {stars}")
                            
                            # Features badges
                            features = []
                            if r.get('query_intent'):
                                features.append(f"Intent: {r['query_intent']}")
                            if r.get('expanded_terms'):
                                features.append("Query Expanded")
                            if r.get('personalized'):
                                features.append("Personalized")
                            
                            if features:
                                st.markdown("".join([f"<span class='feature-badge'>{f}</span>"
                                                    for f in features]), unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Scores
                            if 'hybrid_score' in r:
                                score_cols = st.columns(4)
                                with score_cols[0]:
                                    st.metric("VSM+BM25", f"{r.get('vsm_score', 0):.3f}")
                                with score_cols[1]:
                                    st.metric("Semantic", f"{r.get('semantic_score', 0):.3f}")
                                with score_cols[2]:
                                    st.metric("Hybrid Score", f"{r['hybrid_score']:.3f}")
                                with score_cols[3]:
                                    st.progress(min(r['hybrid_score'], 1.0))
                            
                            # Preview
                            with st.expander("📖 Preview Content"):
                                content = r['document']['content']
                                st.write(content[:500] + "..." if len(content) > 500 else content)
                            
                            # Actions
                            action_cols = st.columns([2, 1, 1])
                            with action_cols[0]:
                                if st.button(f"🔍 Find Similar Books", key=f"similar_{r['document']['id']}"):
                                    similar_results = engine.more_like_this(r['document']['id'], top_k=3)
                                    if similar_results:
                                        st.markdown("**📚 Similar Books:**")
                                        for sim in similar_results:
                                            st.caption(f"• {sim['document']['title']} (similarity: {sim['similarity_score']:.3f})")
                            
                            with action_cols[1]:
                                if st.button("👍 Like", key=f"like_{r['document']['id']}"):
                                    engine.personalization.update_user_profile(
                                        st.session_state.user_id,
                                        r['document']['id'],
                                        'click', 1.0
                                    )

                            with action_cols[2]:
                                if st.button("👎 Dislike", key=f"dislike_{r['document']['id']}"):
                                    engine.personalization.update_user_profile(
                                        st.session_state.user_id,
                                        r['document']['id'],
                                        'skip', -0.5
                                    )
                        
                        # Spacing between cards
                        st.markdown("---")

                                            
                        
                    
                    with tab2:
                        st.markdown("### 📊 Search Analytics")
                        
                        # All analytics code goes here - properly indented
                        scores = [r.get('hybrid_score', r.get('score', 0)) for r in results]
                        
                        # Score Distribution Histogram
                        fig_scores = px.histogram(
                            x=scores, nbins=20,
                            labels={'x': 'Relevance Score', 'y': 'Count'},
                            title='Score Distribution',
                            color_discrete_sequence=['#2e7d32']
                        )
                        fig_scores.update_layout(
                            plot_bgcolor='rgba(255, 255, 255, 0.05)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            font=dict(color='#f9fafb', size=14, family='Inter', weight=600),
                            title_font_size=20,
                            title_font_family='Inter',
                            title_font_color='#d4c57a',
                            title_font_weight=700,
                            xaxis=dict(
                                title='Relevance Score',
                                title_font_size=16,
                                title_font_weight=600,
                                title_font_color='#d4c57a',
                                tickfont_size=14,
                                tickfont_color='#d1d5db',
                                tickfont_family='Inter',
                                showgrid=True,
                                gridcolor='rgba(181, 168, 92, 0.2)',
                                gridwidth=1,
                                zerolinecolor='rgba(181, 168, 92, 0.3)'
                            ),
                            yaxis=dict(
                                title='Count',
                                title_font_size=16,
                                title_font_weight=600,
                                title_font_color='#d4c57a',
                                tickfont_size=14,
                                tickfont_color='#d1d5db',
                                tickfont_family='Inter',
                                showgrid=True,
                                gridcolor='rgba(181, 168, 92, 0.2)',
                                gridwidth=1,
                                zerolinecolor='rgba(181, 168, 92, 0.3)'
                            ),
                            margin=dict(l=70, r=100, t=80, b=70)
                        )
                        fig_scores.update_traces(marker_color='#b5a85c', marker_line_color='#d4c57a', marker_line_width=2)
                        st.plotly_chart(fig_scores, width="stretch")
                        
                        # Score Components Line Chart (only if hybrid scores exist)
                        if 'hybrid_score' in results[0]:
                            df_scores = pd.DataFrame([
                                {
                                    'Rank': r['rank'],
                                    'VSM': r.get('vsm_score', 0),
                                    'BM25': r.get('bm25_score', 0),
                                    'Semantic': r.get('semantic_score', 0),
                                    'Hybrid': r['hybrid_score']
                                } for r in results[:10]
                            ])
                            
                            fig_components = px.line(
                                df_scores, x='Rank',
                                y=['VSM', 'BM25', 'Semantic', 'Hybrid'],
                                title='Score Components by Rank',
                                markers=True,
                                color_discrete_sequence=['#0d3c0f', '#1b5e20', '#2e7d32', '#4CAF50']
                            )
                            fig_components.update_layout(
                                plot_bgcolor='rgba(255, 255, 255, 0.05)',
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                font=dict(color='#f9fafb', size=14, family='Inter', weight=600),
                                title_font_size=20,
                                title_font_family='Inter',
                                title_font_color='#d4c57a',
                                title_font_weight=700,
                                xaxis=dict(
                                    title='Rank',
                                    title_font_size=16,
                                    title_font_weight=600,
                                    title_font_color='#d4c57a',
                                    tickfont_size=14,
                                    tickfont_color='#d1d5db',
                                    tickfont_family='Inter',
                                    showgrid=True,
                                    gridcolor='rgba(181, 168, 92, 0.2)',
                                    gridwidth=1,
                                    zerolinecolor='rgba(181, 168, 92, 0.3)'
                                ),
                                yaxis=dict(
                                    title='Score',
                                    title_font_size=16,
                                    title_font_weight=600,
                                    title_font_color='#d4c57a',
                                    tickfont_size=14,
                                    tickfont_color='#d1d5db',
                                    tickfont_family='Inter',
                                    showgrid=True,
                                    gridcolor='rgba(181, 168, 92, 0.2)',
                                    gridwidth=1,
                                    zerolinecolor='rgba(181, 168, 92, 0.3)'
                                ),
                                legend=dict(
                                    font_size=14,
                                    font_color='#f9fafb',
                                    font_family='Inter',
                                    bgcolor='rgba(181, 168, 92, 0.15)',
                                    bordercolor='rgba(181, 168, 92, 0.4)',
                                    borderwidth=2
                                ),
                                
                            )
                            fig_components.update_traces(line_width=3, marker_size=8)
                            st.plotly_chart(fig_components, width="stretch")
                        
                        # Genre Distribution Pie Chart
                        if facets['genres']:
                            genres_df = pd.DataFrame(
                                facets['genres'].most_common(10),
                                columns=['Genre', 'Count']
                            )
                            fig_genres = px.pie(
                                genres_df, values='Count', names='Genre',
                                title='Genre Distribution in Results',
                                color_discrete_sequence=['#0d3c0f', '#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#4CAF50', '#66bb6a', '#81C784', '#a5d6a7', '#c8e6c9']
                            )
                            fig_genres.update_layout(
                                plot_bgcolor='rgba(255, 255, 255, 0.05)',
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                font=dict(color='#f9fafb', size=14, family='Inter', weight=600),
                                title_font_size=20,
                                title_font_family='Inter',
                                title_font_color='#d4c57a',
                                title_font_weight=700,
                                legend=dict(
                                    font_size=14,
                                    font_color='#f9fafb',
                                    font_family='Inter',
                                    bgcolor='rgba(181, 168, 92, 0.15)',
                                    bordercolor='rgba(181, 168, 92, 0.4)',
                                    borderwidth=2
                                ),
                                margin=dict(l=50, r=100, t=80, b=50)
                            )
                            fig_genres.update_traces(
                                textposition='inside',
                                textfont_size=14,
                                textfont_color='white',
                                textfont_family='Inter',
                                textfont_weight=700,
                                marker=dict(line=dict(color='rgba(181, 168, 92, 0.8)', width=2))
                            )
                            st.plotly_chart(fig_genres, width="stretch")
                
                with tab3:
                    st.markdown("### 🔬 Method Comparison")
                    
                    compare_button = st.button("🔄 Compare All Search Methods", type="primary", width="stretch")
                    
                    if compare_button:
                        with st.spinner("Running comparison across all methods..."):
                            methods = ['VSM', 'Semantic', 'Hybrid', 'Hybrid+QE']
                            comparison_results = {}
                            
                            for method in methods:
                                if method == 'VSM':
                                    res = engine.vsm_search(query, top_k=5)
                                elif method == 'Semantic':
                                    res = engine.semantic_search(query, top_k=5)
                                elif method == 'Hybrid':
                                    res = engine.hybrid_search(query, top_k=5, use_expansion=False)
                                else:
                                    res = engine.hybrid_search(query, top_k=5, use_expansion=True)
                                
                                comparison_results[method] = res
                            
                            st.session_state.compared_results = comparison_results
                    
                    if st.session_state.compared_results:
                        comp_cols = st.columns(len(st.session_state.compared_results))
                        
                        for i, (method, comp_results) in enumerate(st.session_state.compared_results.items()):
                            with comp_cols[i]:
                                st.markdown(f"**{method}**")
                                for j, res in enumerate(comp_results[:3]):
                                    st.caption(f"{j+1}. {res['document']['title'][:30]}...")
                        
                        st.markdown("#### 📊 Result Overlap Analysis")
                        
                        all_methods = list(st.session_state.compared_results.keys())
                        overlap_matrix = []
                        
                        for m1 in all_methods:
                            row = []
                            for m2 in all_methods:
                                ids1 = {r['document']['id'] for r in
                                       st.session_state.compared_results[m1]}
                                ids2 = {r['document']['id'] for r in
                                       st.session_state.compared_results[m2]}
                                overlap = len(ids1 & ids2) / len(ids1 | ids2) if (ids1 | ids2) else 0
                                row.append(overlap)
                            overlap_matrix.append(row)
                        
                        fig_overlap = go.Figure(data=go.Heatmap(
                            z=overlap_matrix,
                            x=all_methods,
                            y=all_methods,
                            colorscale=[[0, '#f1f8e9'], [0.2, '#c8e6c9'], [0.4, '#81C784'], [0.6, '#4CAF50'], [0.8, '#2e7d32'], [1, '#1b5e20']],
                            text=[[f'{val:.2f}' for val in row] for row in overlap_matrix],
                            texttemplate='%{text}',
                            textfont=dict(size=18, color='white', family='Inter', weight=700),
                            showscale=True,
                            colorbar=dict(
                                tickfont=dict(size=14, color='#1a1a1a', family='Inter', weight=600),
                                len=0.7,
                                thickness=20
                            )
                        ))
                        fig_overlap.update_layout(
                            title='Method Result Overlap (Jaccard Similarity)',
                            xaxis_title='Method',
                            yaxis_title='Method',
                            plot_bgcolor='rgba(255, 255, 255, 0.05)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            font=dict(color='#f9fafb', size=14, family='Inter', weight=600),
                            title_font_size=20,
                            title_font_family='Inter',
                            title_font_color='#d4c57a',
                            title_font_weight=700,
                            xaxis=dict(
                                title_font_size=16,
                                title_font_weight=600,
                                title_font_color='#d4c57a',
                                tickfont_size=14,
                                tickfont_color='#d1d5db',
                                tickfont_family='Inter'
                            ),
                            yaxis=dict(
                                title_font_size=16,
                                title_font_weight=600,
                                title_font_color='#d4c57a',
                                tickfont_size=14,
                                tickfont_color='#d1d5db',
                                tickfont_family='Inter'
                            ),
                            margin=dict(l=100, r=150, t=100, b=100)
                        )
                        st.plotly_chart(fig_overlap, width="stretch")
        
        else:
            st.warning("😕 No results found. Try different keywords or broader search terms!")
    
    if hasattr(st.session_state, 'show_analytics') and st.session_state.show_analytics:
        st.markdown("---")
        st.markdown("## 📊 Analytics Dashboard")
        
        if st.session_state.search_history:
            history_df = pd.DataFrame(st.session_state.search_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                fig_timeline = px.scatter(
                    history_df, x='timestamp', y='search_time',
                    size='results_count', hover_data=['query'],
                    title='Search Performance Over Time',
                    color_discrete_sequence=['#2e7d32']
                )
                fig_timeline.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1a1a1a', size=14, family='Inter', weight=600),
                    title_font_size=20,
                    title_font_family='Inter',
                    title_font_color='#1a1a1a',
                    title_font_weight=700,
                    xaxis=dict(
                        title='Time',
                        title_font_size=16,
                        title_font_weight=600,
                        title_font_color='#1a1a1a',
                        tickfont_size=14,
                        tickfont_color='#1a1a1a',
                        tickfont_family='Inter',
                        showgrid=True,
                        gridcolor='#d0d0d0',
                        gridwidth=1
                    ),
                    yaxis=dict(
                        title='Search Time (s)',
                        title_font_size=16,
                        title_font_weight=600,
                        title_font_color='#1a1a1a',
                        tickfont_size=14,
                        tickfont_color='#1a1a1a',
                        tickfont_family='Inter',
                        showgrid=True,
                        gridcolor='#d0d0d0',
                        gridwidth=1
                    ),
                    margin=dict(l=70, r=150, t=80, b=70)
                )
                fig_timeline.update_traces(marker=dict(size=14, line=dict(width=2, color='#1b5e20')))
                st.plotly_chart(fig_timeline, width="stretch")
            
            with col2:
                intent_counts = history_df['intent'].value_counts()
                fig_intent = px.pie(
                    values=intent_counts.values,
                    names=intent_counts.index,
                    title='Query Intent Distribution',
                    color_discrete_sequence=['#0d3c0f', '#1b5e20', '#2e7d32', '#388e3c', '#4CAF50', '#66bb6a']
                )
                fig_intent.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1a1a1a', size=14, family='Inter', weight=600),
                    title_font_size=20,
                    title_font_family='Inter',
                    title_font_color='#1a1a1a',
                    title_font_weight=700,
                    legend=dict(
                        font_size=14,
                        font_color='#1a1a1a',
                        font_family='Inter',
                        bgcolor='rgba(255,255,255,0.95)',
                        bordercolor='#c0c0c0',
                        borderwidth=2
                    ),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                fig_intent.update_traces(textposition='inside', textfont_size=16, textfont_color='white', textfont_family='Inter', textfont_weight=700)
                st.plotly_chart(fig_intent, width="stretch")
            
            st.markdown("### 🔝 Top Queries")
            query_counts = Counter(history_df['query'])
            top_queries_df = pd.DataFrame(
                query_counts.most_common(10),
                columns=['Query', 'Count']
            )
            st.dataframe(top_queries_df, width="stretch")
        
        if st.button("✕ Close Dashboard", width="stretch"):
            del st.session_state.show_analytics
            st.rerun()

# Footer
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 4rem 2rem;'>
    <div style='max-width: 800px; margin: 0 auto;'>
        <h2 style='font-family: "Space Grotesk", serif; color: #d4c57a; font-size: 2rem; margin-bottom: 1rem; font-weight: 600;'>
            E-Library Search Engine
        </h2>
        <p style='color: #b0b0b0; font-size: 1rem; margin-bottom: 1.5rem; line-height: 1.6;'>
            CSD358 - Information Retrieval Project
        </p>
        <p style='color: #b5a85c; font-size: 0.95rem; margin-bottom: 2rem;'>
            Hybrid Search • BM25 Ranking • Query Expansion • Semantic Understanding<br>
            Personalization • Intent Classification • RAG with LLM • A/B Testing
        </p>
        <p style='color: #8a8a8a; font-size: 0.85rem;'>
            © 2025 IR Project Team • Built with Advanced Information Retrieval Techniques
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
