from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# FAQ Data
FAQ_DATA = [
    {"q": "where is your location", 
     "a": "📍 Dubai Digital Park, Silicon Oasis Building A3, Lower Ground. Open daily 10am–10pm."},
    
    {"q": "how to get to your store by metro", 
     "a": "🚇 Take metro to Centrepoint or Al Rashidiya → Bus 320 to Academic City → Silicon Park stop → Walk 5 minutes to Building A3."},
    
    {"q": "can i sell books", 
     "a": "✅ Yes, you can sell your books with us. We upload them to our website and you receive credit or cash once sold."},
    
    {"q": "free delivery", 
     "a": "🚚 Yes, free delivery applies automatically for orders above AED 180."},
    
    {"q": "where is my order", 
     "a": "🔍 Please provide your order number or name so we can check the status."},
    
    {"q": "storage fee", 
     "a": "💰 Storage fee is a one-time charge deducted from your sales."},
    
    {"q": "redemption fee", 
     "a": "💳 10% fee applies only if you convert credit to cash. It is waived if used in-store or online."},
    
    {"q": "how to use credit", 
     "a": "🎯 Use your credit online or in-store and we will deduct it manually."},
    
    {"q": "pick up books", 
     "a": "📦 Pickup is available with AED 25 charge up to 5kg, AED 2 per extra kg."},
    
    {"q": "delivery cost", 
     "a": "💰 AED 19 in Dubai/Sharjah/Ajman and AED 24 in other emirates."},
    
    {"q": "how long books take to sell", 
     "a": "⏰ There is no fixed time. It depends on book popularity and condition."},
    
    {"q": "can i cancel order", 
     "a": "✅ Yes, please tell us the reason for cancellation."},
    
    {"q": "discount code", 
     "a": "🏷️ We currently offer free delivery for orders above AED 180."},
    
    {"q": "operating hours", 
     "a": "🕐 We are open daily from 10am to 10pm."},
    
    {"q": "contact number", 
     "a": "📞 Please contact us through our website or visit our store for assistance."},
    
    {"q": "return policy", 
     "a": "🔄 Books can be returned within 14 days of purchase with original receipt."},
    
    {"q": "gift card", 
     "a": "🎁 Yes, we offer gift cards. Please visit our store for purchase."}
]

@st.cache_resource
def initialize_faq_bot():
    """Initialize FAQ vectorizer"""
    questions = [item["q"] for item in FAQ_DATA]
    vectorizer = TfidfVectorizer()
    faq_matrix = vectorizer.fit_transform(questions)
    return vectorizer, faq_matrix, questions

def get_faq_answer(user_input, vectorizer, faq_matrix, questions):
    """Get answer for FAQ question"""
    if vectorizer is None or faq_matrix is None:
        return "FAQ system is initializing. Please try again in a moment."
    
    try:
        user_vec = vectorizer.transform([user_input.lower()])
        similarity = cosine_similarity(user_vec, faq_matrix)
        index = similarity.argmax()
        
        # Confidence check (0.2 = 20% similarity threshold)
        if similarity[0][index] < 0.2:
            return "🤔 I couldn't find an exact answer. Please contact support or rephrase your question.\n\n💡 Try asking about: location, delivery, selling books, store hours, or returns."
        
        # Find matching answer
        for item in FAQ_DATA:
            if item["q"] == questions[index]:
                confidence = similarity[0][index] * 100
                return f"{item['a']}\n\n*(Confidence: {confidence:.0f}%)*"
        
        return "Answer not found. Please contact support for assistance."
    except Exception as e:
        return f"Error processing your question: {str(e)}"
