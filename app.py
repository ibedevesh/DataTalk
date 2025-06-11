import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Set page config
st.set_page_config(
    page_title="Atlys Review Analysis",
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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize Gemini client using Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error("Error: Unable to initialize Gemini API. Please check your API key in Streamlit secrets.")
    st.stop()

# Load and process the data
@st.cache_data
def load_data():
    try:
        # Use the raw GitHub URL for the CSV file
        github_url = "https://raw.githubusercontent.com/your-username/your-repo/main/all_reviews.csv"
        df = pd.read_csv(github_url)
        # Clean the data
        df['text'] = df['text'].str.replace('more_vert Flag inappropriate', '')
        df['text'] = df['text'].str.replace('Did you find this helpful?YesNo', '')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_gemini_response(prompt, reviews_df):
    try:
        # Create a context from the reviews (limit to most recent 100 reviews to avoid token limits)
        recent_reviews = reviews_df.tail(100)
        reviews_text = "\n".join([
            f"Rating: {row['rating']}/5\nReview: {row['text']}\n"
            for _, row in recent_reviews.iterrows()
        ])
        
        # Create the full prompt for Gemini
        full_prompt = f"""Here are some recent customer reviews for a visa application service called Atlys:

{reviews_text}

Based on these reviews, please answer the following question:
{prompt}

Please provide a detailed analysis based on the reviews, including specific examples and patterns you notice. Focus on being helpful and constructive in your analysis."""

        # Get response from Gemini
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error getting response from Gemini: {str(e)}"

# Main app layout
st.title("📊 Atlys Review Analysis Chat")
st.write("Ask questions about customer reviews for Atlys visa services")

# Load the data
reviews_df = load_data()
if reviews_df is None:
    st.error("Failed to load data. Please check the data source and try again.")
    st.stop()

# Create two columns for the layout
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    user_question = st.text_input("What would you like to know about the reviews?", 
                                placeholder="e.g., What are the main complaints from customers?")

    if user_question:
        with st.spinner("Analyzing reviews..."):
            response = get_gemini_response(user_question, reviews_df)
            st.session_state.chat_history.append({"question": user_question, "answer": response})
            
            # Display the response
            st.markdown("### Analysis")
            st.write(response)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Previous Questions")
        for chat in st.session_state.chat_history:
            with st.expander(chat["question"]):
                st.write(chat["answer"])

with col2:
    # Statistics
    st.markdown("### 📈 Review Statistics")
    total_reviews = len(reviews_df)
    avg_rating = reviews_df['rating'].mean()
    
    st.metric("Total Reviews", total_reviews)
    st.metric("Average Rating", f"{avg_rating:.2f}/5")
    
    # Rating distribution
    st.markdown("### Rating Distribution")
    rating_counts = reviews_df['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What are the main complaints from customers?",
        "What aspects of the service do customers praise?",
        "How do customers rate the customer support?",
        "What are the common issues with visa processing?",
        "What is the overall customer satisfaction level?",
        "What are the most common positive experiences?",
        "How do customers feel about the visa processing time?",
        "What improvements do customers suggest?"
    ]
    
    for question in example_questions:
        if st.button(question, key=question):
            st.session_state.chat_history.append({
                "question": question,
                "answer": get_gemini_response(question, reviews_df)
            })
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Gemini AI") 
