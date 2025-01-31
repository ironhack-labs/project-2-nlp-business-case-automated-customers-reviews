import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import gc

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Reseñas de Amazon",
    page_icon="📊",
    layout="wide"
)

# Descargar recursos de NLTK necesarios
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        st.warning(f"Error descargando recursos NLTK: {str(e)}")

# Preprocesamiento de texto
@st.cache_data
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        st.warning(f"Error en preprocesamiento: {str(e)}")
        return text

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed_reviews.csv')
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None

def main():
    st.title("📊 Análisis de Reseñas de Amazon")
    
    # Cargar recursos
    with st.spinner('Cargando recursos necesarios...'):
        download_nltk_resources()
        df = load_data()
    
    if df is not None:
        # Sidebar para filtros
        st.sidebar.header("Filtros")
        
        # Filtro de categoría
        categories = ['Todas'] + sorted(df['product_category'].unique().tolist())
        selected_category = st.sidebar.selectbox(
            "Categoría de Producto",
            categories
        )
        
        # Aplicar filtros
        if selected_category != 'Todas':
            df_filtered = df[df['product_category'] == selected_category]
        else:
            df_filtered = df
        
        # Layout en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de sentimientos
            sentiment_counts = df_filtered['sentiment'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribución de Sentimientos",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # Distribución de ratings
            rating_counts = df_filtered['star_rating'].value_counts().sort_index()
            fig_ratings = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title="Distribución de Ratings",
                labels={'x': 'Rating', 'y': 'Cantidad'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
        
        # Resumen de datos
        st.header("📈 Resumen de Datos")
        if st.checkbox("Mostrar datos de ejemplo"):
            st.dataframe(
                df_filtered[['product_category', 'star_rating', 'sentiment', 'cleaned_review']]
                .head(10)
            )
        
        # Métricas generales
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Reseñas", f"{len(df_filtered):,}")
        col2.metric(
            "Rating Promedio",
            f"{df_filtered['star_rating'].mean():.2f}⭐"
        )
        col3.metric(
            "% Positivas",
            f"{(df_filtered['sentiment'] == 'positive').mean():.1%}"
        )
        
        # Análisis por categoría
        if selected_category == 'Todas':
            st.header("📊 Análisis por Categoría")
            category_stats = df.groupby('product_category').agg({
                'star_rating': 'mean',
                'sentiment': lambda x: (x == 'positive').mean()
            }).round(3)
            
            category_stats.columns = ['Rating Promedio', '% Positivas']
            st.dataframe(category_stats)
    
    else:
        st.warning("No se pudieron cargar los datos. Por favor, verifica la instalación.")

if __name__ == "__main__":
    try:
        main()
    finally:
        gc.collect()