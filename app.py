import streamlit as st
import pandas as pd
from anthropic import Anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Load and process the data
@st.cache_data
def load_data():
    df = pd.read_csv('all_reviews.csv')
    return df

def get_claude_response(prompt, reviews_df):
    # Create a context from the reviews
    reviews_text = "\n".join([
        f"Rating: {row['rating']}/5\nReview: {row['text']}\n"
        for _, row in reviews_df.iterrows()
    ])
    
    # Create the full prompt for Claude
    full_prompt = f"""Here are some customer reviews for a visa application service called Atlys:

{reviews_text}

Based on these reviews, please answer the following question:
{prompt}

Please provide a detailed analysis based on the reviews, including specific examples and patterns you notice."""

    # Get response from Claude
    response = client.messages.create(
        model=os.getenv('CLAUDE_MODEL_NAME', 'claude-3-sonnet-20240229'),
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": full_prompt
        }]
    )
    
    return response.content[0].text

# Set up the Streamlit app
st.title("Atlys Review Analysis Chat")
st.write("Ask questions about customer reviews for Atlys visa services")

# Load the data
reviews_df = load_data()

# Create a chat interface
user_question = st.text_input("What would you like to know about the reviews?")

if user_question:
    with st.spinner("Analyzing reviews..."):
        response = get_claude_response(user_question, reviews_df)
        st.write("Analysis:")
        st.write(response)

# Add some helpful example questions
st.sidebar.title("Example Questions")
st.sidebar.write("Try asking:")
st.sidebar.write("- What are the main complaints from customers?")
st.sidebar.write("- What aspects of the service do customers praise?")
st.sidebar.write("- How do customers rate the customer support?")
st.sidebar.write("- What are the common issues with visa processing?")
st.sidebar.write("- What is the overall customer satisfaction level?")

# Add some statistics
st.sidebar.title("Review Statistics")
st.sidebar.write(f"Total Reviews: {len(reviews_df)}")
st.sidebar.write(f"Average Rating: {reviews_df['rating'].mean():.2f}/5")
st.sidebar.write(f"5-star Reviews: {len(reviews_df[reviews_df['rating'] == 5])}")
st.sidebar.write(f"1-star Reviews: {len(reviews_df[reviews_df['rating'] == 1])}") 
