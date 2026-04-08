import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_resource
def initialize_recommender(books):
    """Initialize TF-IDF vectorizer and similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim

def recommend_by_title(title, books, cosine_sim, top_n=5):
    """Recommend books based on title similarity"""
    if cosine_sim is None:
        return ["Recommendation system not initialized. Please refresh the page."]
    
    if title not in books['Book Title'].values:
        return [f"❌ '{title}' not found. Please try another title."]
    
    try:
        idx = books[books['Book Title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for i in sim_scores:
            book_title = books['Book Title'].iloc[i[0]]
            author = books['Author'].iloc[i[0]]
            similarity_score = i[1] * 100
            recommendations.append(f"📖 **{book_title}** by *{author}* (Match: {similarity_score:.1f}%)")
        
        return recommendations
    except Exception as e:
        return [f"Error generating recommendations: {str(e)}"]

def recommend_by_genre(genre, books, cosine_sim=None, top_n=5):
    """Recommend books based on genre"""
    genre = genre.lower()
    filtered = books[books['Genre'].str.contains(genre, na=False)]
    
    if filtered.empty:
        return [f"❌ No books found in '{genre}' genre. Try: self-help, finance, productivity, business, history, memoir, fiction"]
    
    # Take sample or top books
    if len(filtered) > top_n:
        filtered = filtered.sample(min(top_n, len(filtered)))
    else:
        filtered = filtered.head(top_n)
    
    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append(f"📖 **{row['Book Title']}** by *{row['Author']}*")
    
    return recommendations

def vibe_recommend(text, books, top_n=5):
    """Recommend books based on vibe/text search"""
    text = text.lower()
    
    # Search in combined features
    filtered = books[books['combined'].str.lower().str.contains(text, na=False)]
    
    if filtered.empty:
        # If no exact match, try to match individual words
        keywords = text.split()
        mask = pd.Series([False] * len(books))
        for keyword in keywords:
            if len(keyword) > 2:  # Ignore very short words
                mask = mask | books['combined'].str.lower().str.contains(keyword, na=False)
        filtered = books[mask]
    
    if filtered.empty:
        return [f"❌ No books found matching '{text}'. Try different keywords or use genre search."]
    
    if len(filtered) > top_n:
        filtered = filtered.sample(min(top_n, len(filtered)))
    
    recommendations = []
    for _, row in filtered.iterrows():
        recommendations.append(f"📖 **{row['Book Title']}** by *{row['Author']}*")
    
    return recommendations
