import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px

# Basic page config
st.set_page_config(
    page_title="Review Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Gemini
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error("Error: Unable to initialize Gemini API. Please check your API key in Streamlit secrets.")
    st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('all_reviews.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def analyze_reviews(prompt, reviews_df):
    try:
        recent_reviews = reviews_df.tail(100)
        reviews_text = "\n".join([
            f"Rating: {row['rating']}/5\nReview: {row['text']}\n"
            for _, row in recent_reviews.iterrows()
        ])
        
        full_prompt = f"""Here are some recent customer reviews for a visa application service called Atlys:

{reviews_text}

Based on these reviews, please answer the following question:
{prompt}

Please provide a detailed analysis based on the reviews, including specific examples and patterns you notice."""

        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing reviews: {str(e)}"

# Main interface
st.title("📊 Review Analysis")

# Load data
reviews_df = load_data()
if reviews_df is None:
    st.error("Failed to load data. Please check the data source and try again.")
    st.stop()

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Question input
    user_question = st.text_input("Ask a question about the reviews:", 
                                placeholder="e.g., What are the main complaints from customers?")

    if user_question:
        with st.spinner("Analyzing..."):
            analysis = analyze_reviews(user_question, reviews_df)
            st.markdown("### Analysis")
            st.write(analysis)

with col2:
    # Statistics
    st.markdown("### 📈 Statistics")
    total_reviews = len(reviews_df)
    avg_rating = reviews_df['rating'].mean()
    
    st.metric("Total Reviews", total_reviews)
    st.metric("Average Rating", f"{avg_rating:.2f}/5")
    
    # Rating distribution
    st.markdown("### Rating Distribution")
    rating_counts = reviews_df['rating'].value_counts().sort_index()
    
    # Create the bar chart using plotly
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating', 'y': 'Count'},
        title='Rating Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What are the main complaints from customers?",
        "What aspects of the service do customers praise?",
        "How do customers rate the customer support?",
        "What are the common issues with visa processing?",
        "What is the overall customer satisfaction level?"
    ]
    
    for question in example_questions:
        if st.button(question, key=question):
            with st.spinner("Analyzing..."):
                analysis = analyze_reviews(question, reviews_df)
                st.markdown("### Analysis")
                st.write(analysis)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Gemini AI") 
