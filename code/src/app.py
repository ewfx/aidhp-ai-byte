import json
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import implicit
import numpy as np
from scipy.sparse import coo_matrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ["OPENAI_API_KEY"]

# Initialize FastAPI app
app = FastAPI()

# Load JSON data
with open("data/customer_profiles_filled_100.json", "r") as f:
    customer_profiles = json.load(f)
with open("data/transaction_history_filled_100.json", "r") as f:
    transaction_history = json.load(f)
with open("data/social_media_posts_filled_100.json", "r") as f:
    social_media_posts = json.load(f)

# Convert data to DataFrames
transactions_df = pd.DataFrame(transaction_history)
social_media_df = pd.DataFrame(social_media_posts)
customer_profiles_df = pd.DataFrame(customer_profiles)

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return round(analyzer.polarity_scores(text)["compound"], 2)

# Apply sentiment analysis
social_media_df["sentiment_score"] = social_media_df["content"].apply(get_sentiment)

# Normalize sentiment scores
social_media_df["sentiment_preference"] = (social_media_df["sentiment_score"] + 1) / 2

# Intent classification model
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
intent_labels = ["Fashion", "Shopping", "Budget Concern", "Tech Enthusiast", "Fitness", "Finance", "Travel", "Tech Review"]

def get_intent(text):
    prediction = intent_classifier(text, intent_labels)
    return prediction["labels"][0]

social_media_df["intent"] = social_media_df["content"].apply(get_intent)

def get_top_transaction_categories(customer_id, top_n=3):
    user_transactions = transactions_df[transactions_df["customer_id"] == customer_id]
    return ", ".join([cat for cat, _ in Counter(user_transactions["category"]).most_common(top_n)])

def get_top_social_media_intents(customer_id, top_n=3):
    user_posts = social_media_df[social_media_df["customer_id"] == customer_id]
    return ", ".join([intent for intent, _ in Counter(user_posts["intent"]).most_common(top_n)])


# OpenAI API Response Generator
def generate_response(prompt):
    try:
        client = openai.OpenAI()  # Create an OpenAI client instance

        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
        return response.choices[0].message.content
    except openai.error.OpenAIError as e:
        return f"OpenAI API Error: {e}"

# API Endpoint for Business Insights
@app.get("/business_insights")
def generate_business_strategy():
    top_purchases = transactions_df["category"].value_counts().nlargest(3).index.tolist()
    avg_sentiment = social_media_df["sentiment_score"].mean()
    sentiment_trend = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"
    
    prompt = f"""
        As a banking institution, analyze customer transactions and sentiment data to identify financial opportunities.

        ### **Customer Insights:**
        - **Top 3 Financial Services in Demand:** {', '.join(top_purchases)}
        - **Overall Customer Sentiment:** {sentiment_trend}

        ### **Hyper-Personalized Financial Product Recommendations:**
        Provide **tailored recommendations** based on customer spending patterns, preferences, and financial behavior.

        #### **1️ Credit Card Recommendations**
        Identify suitable credit cards based on:
        - Spending habits (e.g., travel, dining, shopping, fuel, groceries).
        - Rewards interests (e.g., cashback, airline miles, dining discounts).
        - Financial profile.

        **Example:** If a customer frequently spends on travel and dining, recommend a **premium travel rewards credit card**.

        #### **2 Loan & Mortgage Suggestions**
        Recommend suitable loan options considering:
        - **High credit score & stable income** → Suggest **low-interest loans** (home, car, or personal).
        - **Business owners** → Recommend a **small business loan** with flexible repayment terms.

        #### **3 Investment Portfolio Recommendations**
        Suggest investment strategies based on:
        - Financial goals & risk appetite.
        - Transaction history & sentiment analysis.

        **Example:**  
        - If a customer prefers **low-risk investments**, recommend **index funds or fixed deposits**.  
        - If a customer is willing to take **moderate risks**, suggest a **diversified mutual fund portfolio**.  

        #### **4 Insurance & Wealth Management**
        Recommend **insurance and wealth protection plans** based on customer needs:
        - If the customer has **dependents/family**, suggest a **life insurance plan**.
        - If the customer **travels frequently**, recommend a **travel insurance policy**.
        - For **high-net-worth individuals**, suggest **wealth management services**.

        #### **5 Hyper-Personalized Savings & Retirement Plans**
        Suggest financial security strategies based on:
        - Long-term savings goals.
        - Retirement planning preferences.

        **Example:**  
        - If a customer expresses interest in **financial security**, recommend a **high-interest savings account or Roth IRA**.  
        - For **salaried employees**, suggest **401(k) plans with employer matching**.

        ### **Deliverables:**
        - Provide **data-driven justifications** for each recommendation.
        - Use **multi-modal insights** (transactions, sentiment, and spending behavior).
        - Structure the response clearly with actionable recommendations.

        """
    return {"strategy": generate_response(prompt)}

# API Endpoint for Personalized Recommendation
@app.get("/recommend/{customer_id}")
def generate_personalized_recommendation(customer_id: str):

    if customer_id not in customer_profiles_df['customer_id'].values:
        return "Customer ID not found."
    
    customer = customer_profiles_df[customer_profiles_df["customer_id"] == customer_id].to_dict()
    purchased_categories = transactions_df[transactions_df["customer_id"] == customer_id]["category"].value_counts().index.tolist()
    sentiment_summary = social_media_df[social_media_df["customer_id"] == customer_id]["sentiment_score"].mean()
    sentiment_trend = "Positive" if sentiment_summary > 0.2 else "Negative" if sentiment_summary < -0.2 else "Neutral"
    top_intents = social_media_df[social_media_df["customer_id"] == customer_id]["intent"].value_counts().index.tolist()[:3]
    
    prompt = f"""
    Customer Profile: {customer}
    Purchased Categories: {', '.join(purchased_categories) if purchased_categories else 'No past purchases'}
    Overall Sentiment: {sentiment_trend}
    Social Media Intents: {', '.join(top_intents) if top_intents else 'No social media data'}
    Recommend hyper-personalized financial products in the following categories:
    - Credit Cards
    - Loans & Mortgages
    - Investments (Stocks, Mutual Funds, ETFs)
    - Insurance (Health, Life, Travel)
    - Retirement & Savings Plans

    Provide reasons for each recommendation.
    """

    
    return {"recommendation": generate_response(prompt)}

# Run FastAPI (for local development, use: `uvicorn filename:app --reload`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
