import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utilities from utils folder
from utils.data_loader import load_books_data, load_products_data
from utils.recommender import initialize_recommender, recommend_by_title, recommend_by_genre, vibe_recommend
from utils.faq_bot import initialize_faq_bot, get_faq_answer

# Page configuration
st.set_page_config(
    page_title="Bookends - AI Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: white;
        opacity: 0.9;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_resource
def load_all_data():
    """Load all data with caching"""
    try:
        books = load_books_data()
        products = load_products_data()
        
        # Merge with sales data if available
        if not products.empty and 'book title' in products.columns:
            products = products.rename(columns={'book title': 'Book Title'})
            books = books.merge(products, on='Book Title', how='left')
            if 'total sales' in books.columns:
                books['total sales'] = books['total sales'].fillna(0)
        
        return books
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Initialize systems
def init_systems(books):
    """Initialize recommendation and FAQ systems"""
    if books is not None and not books.empty:
        tfidf, tfidf_matrix, cosine_sim = initialize_recommender(books)
        faq_vectorizer, faq_matrix, faq_questions = initialize_faq_bot()
        return cosine_sim, faq_vectorizer, faq_matrix, faq_questions
    return None, None, None, None

# Load everything
books = load_all_data()
data_loaded = books is not None and not books.empty

if data_loaded:
    cosine_sim, faq_vectorizer, faq_matrix, faq_questions = init_systems(books)
else:
    cosine_sim = faq_vectorizer = faq_matrix = faq_questions = None

# Header
st.markdown("""
    <div class="main-header">
        <h1>📚 Bookends AI Book Recommendation System</h1>
        <p>Discover your next favorite book with AI-powered recommendations</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose a feature:",
    ["📖 Book Recommender", "💬 FAQ Chatbot", "📊 Dashboard", "ℹ️ About"]
)

# Book Recommender
if menu == "📖 Book Recommender":
    st.header("📖 Find Your Next Book")
    
    if not data_loaded:
        st.warning("⚠️ Data not loaded. Using sample data for demonstration.")
        # Create sample data for demo
        sample_books = pd.DataFrame({
            'Book Title': ['Atomic Habits', 'The Psychology of Money', 'Deep Work', 
                          'The 7 Habits of Highly Effective People', 'Think and Grow Rich',
                          'Rich Dad Poor Dad', 'The Intelligent Investor', 'Zero to One'],
            'Author': ['James Clear', 'Morgan Housel', 'Cal Newport', 
                      'Stephen R. Covey', 'Napoleon Hill', 'Robert Kiyosaki',
                      'Benjamin Graham', 'Peter Thiel'],
            'Genre': ['self-help', 'finance', 'productivity', 
                     'self-help', 'self-help', 'finance', 'finance', 'business']
        })
        books = sample_books
        books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(books['combined'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        recommendation_type = st.selectbox(
            "How would you like to find books?",
            ["By Genre", "By Book Title", "By Vibe/Description"]
        )
    
    with col2:
        if recommendation_type == "By Genre":
            genre = st.text_input("Enter a genre (e.g., kids, romance, history, self-help, finance):", 
                                 placeholder="e.g., romance")
            if st.button("🔍 Find Books", type="primary"):
                if genre:
                    with st.spinner("Finding recommendations..."):
                        recommendations = recommend_by_genre(genre, books, cosine_sim if 'cosine_sim' in locals() else None)
                        st.subheader(f"📚 {recommendation_type} Results:")
                        for rec in recommendations:
                            st.markdown(f'<div class="recommendation-card">{rec}</div>', 
                                      unsafe_allow_html=True)
                else:
                    st.warning("Please enter a genre.")
        
        elif recommendation_type == "By Book Title":
            # Get list of available titles
            book_titles = books['Book Title'].tolist()
            title = st.selectbox("Select a book you like:", book_titles)
            if st.button("🔍 Find Similar Books", type="primary"):
                with st.spinner("Finding similar books..."):
                    recommendations = recommend_by_title(title, books, cosine_sim if 'cosine_sim' in locals() else None)
                    st.subheader(f"📚 Books similar to '{title}':")
                    for rec in recommendations:
                        st.markdown(f'<div class="recommendation-card">{rec}</div>', 
                                  unsafe_allow_html=True)
        
        else:  # By Vibe
            vibe = st.text_area("Describe what you're looking for (e.g., 'inspiring stories about success', 'romantic comedies'):",
                               placeholder="e.g., books that make you think about success")
            if st.button("🔍 Find Books", type="primary"):
                if vibe:
                    with st.spinner("Finding books matching your vibe..."):
                        recommendations = vibe_recommend(vibe, books)
                        st.subheader(f"📚 Recommendations based on your vibe:")
                        for rec in recommendations:
                            st.markdown(f'<div class="recommendation-card">{rec}</div>', 
                                      unsafe_allow_html=True)
                else:
                    st.warning("Please describe what you're looking for.")

# FAQ Chatbot
elif menu == "💬 FAQ Chatbot":
    st.header("💬 Bookends FAQ Assistant")
    st.markdown("Ask me anything about orders, delivery, selling books, or store policies!")
    
    # Example questions
    with st.expander("📝 Example questions you can ask:"):
        st.markdown("""
        - Where is your location?
        - Can I sell books?
        - Do you offer free delivery?
        - How do I use my credit?
        - What is the delivery cost?
        - Can I cancel my order?
        - What are your operating hours?
        """)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if faq_vectorizer is not None:
                    response = get_faq_answer(prompt, faq_vectorizer, faq_matrix, faq_questions)
                else:
                    response = "FAQ system is initializing. Please try again in a moment."
                st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Dashboard
elif menu == "📊 Dashboard":
    st.header("📊 Insights Dashboard")
    
    if data_loaded and books is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Genres")
            genre_counts = books['Genre'].value_counts().head(10)
            if not genre_counts.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                genre_counts.plot(kind='barh', ax=ax, color='skyblue')
                ax.set_xlabel("Number of Books")
                ax.set_title("Most Popular Genres")
                st.pyplot(fig)
            else:
                st.info("No genre data available")
        
        with col2:
            st.subheader("Genre Distribution")
            genre_counts = books['Genre'].value_counts().head(8)
            if not genre_counts.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                genre_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
                ax2.set_ylabel("")
                st.pyplot(fig2)
            else:
                st.info("No genre data available")
        
        # Statistics
        st.subheader("📈 Statistics")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Total Books", len(books))
        with col4:
            st.metric("Unique Genres", books['Genre'].nunique())
        with col5:
            st.metric("Unique Authors", books['Author'].nunique())
        
        # Top Authors
        st.subheader("Top Authors by Book Count")
        top_authors = books['Author'].value_counts().head(10)
        if not top_authors.empty:
            st.bar_chart(top_authors)
        else:
            st.info("No author data available")
        
    else:
        st.info("📊 Dashboard will be available once data is loaded. Using sample data for demonstration.")
        # Show sample dashboard with demo data
        sample_data = pd.DataFrame({
            'Genre': ['Self-help', 'Finance', 'Productivity', 'Business', 'Technology'],
            'Count': [25, 18, 15, 12, 10]
        })
        st.bar_chart(sample_data.set_index('Genre'))

# About page
elif menu == "ℹ️ About":
    st.header("ℹ️ About Bookends AI System")
    
    st.markdown("""
    ### 📚 **Welcome to Bookends AI Book Recommendation System**
    
    This intelligent system helps you discover books based on:
    - **Genre preferences**
    - **Similar titles**
    - **Vibe/description matching**
    
    ### 🤖 **Features**
    
    - **Smart Recommendations**: Using TF-IDF and cosine similarity algorithms
    - **FAQ Chatbot**: Instant answers to common questions about our services
    - **Interactive Dashboard**: Visual insights about our book collection
    - **User-friendly Interface**: Simple and intuitive design for easy navigation
    
    ### 🛠️ **Technologies Used**
    
    - **Frontend**: Streamlit - Powerful Python web framework
    - **Backend**: Python 3.9+ - Core programming language
    - **ML/NLP**: Scikit-learn, TF-IDF for text analysis
    - **Data Processing**: Pandas, NumPy for efficient data handling
    - **Visualization**: Matplotlib, Seaborn for beautiful charts
    
    ### 📞 **Contact & Support**
    
    For questions or support:
    - **Visit us**: Dubai Digital Park, Silicon Oasis Building A3
    - **Email**: support@bookends.ae
    - **Phone**: Available in-store
    
    ### 📅 **Store Hours**
    **Daily**: 10:00 AM - 10:00 PM
    
    ### 🚚 **Delivery Information**
    - **Free delivery** on orders above AED 180
    - **AED 19** delivery in Dubai/Sharjah/Ajman
    - **AED 24** in other emirates
    - **Pickup option** available with nominal fee
    
    ### 💡 **How It Works**
    
    1. **Content-Based Filtering**: Our system analyzes book features (title, author, genre)
    2. **TF-IDF Vectorization**: Converts text into numerical features
    3. **Cosine Similarity**: Finds books most similar to your preferences
    4. **Smart Ranking**: Presents recommendations in order of relevance
    
    ### 🎯 **Use Cases**
    
    - Discover new books based on your favorites
    - Explore genres you might enjoy
    - Find books matching specific themes or vibes
    - Get instant answers to store-related questions
    
    ---
    
    *Made with ❤️ for book lovers in the UAE and beyond*
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Bookends AI System** v2.0
    
    Made with ❤️ for book lovers
    
    ---
    *💡 Tip: Ask our FAQ chatbot for instant answers!*
    """
)

# Display data status in sidebar
if data_loaded:
    st.sidebar.success("✅ System Ready")
    st.sidebar.caption(f"📚 {len(books)} books available")
else:
    st.sidebar.warning("⚠️ Using demo mode")
