import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_books_data():
    """Load and cache books data"""
    try:
        # Try multiple possible paths
        possible_paths = [
            "data/DetailedBooksExcel Cleaned (RemovedBlank).xlsx",
            "DetailedBooksExcel Cleaned (RemovedBlank).xlsx",
            "../data/DetailedBooksExcel Cleaned (RemovedBlank).xlsx"
        ]
        
        books = None
        for path in possible_paths:
            if os.path.exists(path):
                books = pd.read_excel(path)
                break
        
        if books is None:
            # Return sample data if file not found
            return create_sample_books_data()
        
        books = books.dropna(subset=['Book Title', 'Author', 'Genre'])
        books['Book Title'] = books['Book Title'].astype(str)
        books['Author'] = books['Author'].astype(str)
        books['Genre'] = books['Genre'].astype(str).str.lower()
        
        books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
        
        return books
    except Exception as e:
        st.warning(f"Could not load books data: {e}. Using sample data.")
        return create_sample_books_data()

@st.cache_data
def load_products_data():
    """Load and cache products data"""
    try:
        possible_paths = [
            "data/Total sales by product - 2026-01-31 - 2026-03-02 (Cleaned).csv",
            "Total sales by product - 2026-01-31 - 2026-03-02 (Cleaned).csv",
            "data/Total sales by product Clean.csv",
            "Total sales by product Clean.csv"
        ]
        
        products_list = []
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                products_list.append(df)
        
        if products_list:
            products = pd.concat(products_list)
            products.columns = products.columns.str.strip().str.lower()
            return products
        
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def create_sample_books_data():
    """Create sample data for testing/demo"""
    sample_data = {
        'Book Title': [
            'Atomic Habits', 'The Psychology of Money', 'Deep Work', 
            'The 7 Habits of Highly Effective People', 'Think and Grow Rich',
            'Rich Dad Poor Dad', 'The Intelligent Investor', 'Zero to One',
            'Sapiens', 'Becoming', 'Educated', 'The Alchemist'
        ],
        'Author': [
            'James Clear', 'Morgan Housel', 'Cal Newport', 
            'Stephen R. Covey', 'Napoleon Hill', 'Robert Kiyosaki',
            'Benjamin Graham', 'Peter Thiel', 'Yuval Noah Harari',
            'Michelle Obama', 'Tara Westover', 'Paulo Coelho'
        ],
        'Genre': [
            'self-help', 'finance', 'productivity', 
            'self-help', 'self-help', 'finance', 
            'finance', 'business', 'history',
            'memoir', 'memoir', 'fiction'
        ]
    }
    books = pd.DataFrame(sample_data)
    books['combined'] = books['Book Title'] + " " + books['Author'] + " " + books['Genre']
    return books
