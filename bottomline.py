# bottomline.py

import os
import pandas as pd
import joblib
import openai
import gradio as gr
from typing import List, Optional

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Utility Functions
def load_data(file_obj) -> pd.DataFrame:
    """Load dataset from an uploaded file."""
    try:
        return pd.read_csv(file_obj)
    except Exception as e:
        return f"Error loading dataset: {e}"

def load_model(file_obj) -> Optional[object]:
    """Load the machine learning model."""
    try:
        return joblib.load(file_obj)
    except Exception as e:
        return f"Error loading model: {e}"

def load_vectorizer(file_obj) -> Optional[object]:
    """Load the vectorizer."""
    try:
        return joblib.load(file_obj)
    except Exception as e:
        return f"Error loading vectorizer: {e}"

def process_reviews(df: pd.DataFrame, vectorizer, model) -> pd.DataFrame:
    """Vectorize reviews and predict sentiment."""
    try:
        X = vectorizer.transform(df['reviews.text'])
        df['sentiment'] = model.predict(X)
        df['sentiment_score'] = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})
        return df
    except Exception as e:
        return f"Error processing reviews: {e}"

def calculate_average_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average sentiment scores grouped by cluster and product name."""
    try:
        product_scores = df.groupby(['cluster_name', 'name'])['sentiment_score'].mean().reset_index()
        return df.merge(product_scores, on=['cluster_name', 'name'], suffixes=('', '_avg'))
    except Exception as e:
        return f"Error calculating average scores: {e}"

def generate_article(category_name: str, top_products: List[str], bottom_product: str, reviews: str) -> str:
    """Generate an article using OpenAI's ChatCompletion API."""
    messages = [
        {"role": "system", "content": "You are a helpful and engaging blog post writer."},
        {"role": "user", "content": f"""
        Write a blog post titled 'Top Products in {category_name}'.
        Include the following:
        - Introduction to the category.
        - Top 3 products: {', '.join(top_products)}.
          - Key differences between them.
          - When a consumer should choose one over another.
        - Common complaints or issues for each of these top products.
        - The worst product in the category: {bottom_product}.
          - Reasons why consumers should avoid it.
        - Conclusion with a recommendation.
        
        Use a friendly and informative tone, similar to articles on The Verge or Wirecutter.
        Incorporate insights from the following reviews:
        {reviews}
        """}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=1.3,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating article: {e}"

# Gradio App
def gradio_app(data_file, model_file, vectorizer_file):
    """Main function to process data and generate articles."""
    # Load files
    df = load_data(data_file)
    model = load_model(model_file)
    vectorizer = load_vectorizer(vectorizer_file)
    
    # Validate loaded objects
    if isinstance(df, str) or isinstance(model, str) or isinstance(vectorizer, str):
        return f"Error loading files: {df if isinstance(df, str) else ''} {model if isinstance(model, str) else ''} {vectorizer if isinstance(vectorizer, str) else ''}"

    # Process reviews and calculate scores
    processed_df = process_reviews(df, vectorizer, model)
    if isinstance(processed_df, str):
        return processed_df
    scored_df = calculate_average_scores(processed_df)
    if isinstance(scored_df, str):
        return scored_df

    # Generate articles
    articles = []
    for category in scored_df['cluster_name'].unique():
        category_df = scored_df[scored_df['cluster_name'] == category]
        product_sentiments = (
            category_df.groupby('name')['sentiment_score_avg']
            .mean().reset_index()
        )
        top_products = product_sentiments.nlargest(3, 'sentiment_score_avg')['name'].tolist()
        bottom_product = product_sentiments.nsmallest(1, 'sentiment_score_avg')['name'].iloc[0]
        top_reviews = category_df[category_df['name'].isin(top_products)]['reviews.text'].sample(n=5, random_state=42).tolist()

        article = generate_article(
            category,
            top_products,
            bottom_product,
            ' '.join(top_reviews)
        )
        articles.append(f"Category: {category}\n\n{article}\n\n{'-'*50}")
    
    return "\n\n".join(articles)

# Define Gradio Interface
interface = gr.Interface(
    fn=gradio_app,
    inputs=[
        gr.inputs.File(label="Upload Dataset (CSV)"),
        gr.inputs.File(label="Upload Model (Pickle File)"),
        gr.inputs.File(label="Upload Vectorizer (Pickle File)")
    ],
    outputs=gr.outputs.Textbox(label="Generated Articles"),
    title="Bottomline",
    description="Upload the required files to analyze sentiment and generate articles."
)

# Launch Gradio App
if __name__ == "__main__":
    interface.launch()
