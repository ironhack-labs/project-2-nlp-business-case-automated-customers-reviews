"""
Módulo para preprocesamiento de texto de las reseñas de Amazon.
"""

import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, language='english'):
        """
        Inicializa el preprocesador de texto.
        
        Args:
            language (str): Idioma para stopwords ('english' o 'spanish')
        """
        # Descargar recursos necesarios de NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Limpia el texto eliminando caracteres especiales y normalizando.
        
        Args:
            text (str): Texto a limpiar
            
        Returns:
            str: Texto limpio
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Normalizar espacios
        text = ' '.join(text.split())
        
        return text

    def remove_stopwords(self, text):
        """
        Elimina stopwords del texto.
        
        Args:
            text (str): Texto a procesar
            
        Returns:
            str: Texto sin stopwords
        """
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)

    def lemmatize_text(self, text):
        """
        Aplica lematización al texto.
        
        Args:
            text (str): Texto a lematizar
            
        Returns:
            str: Texto lematizado
        """
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def preprocess(self, text):
        """
        Aplica todo el pipeline de preprocesamiento.
        
        Args:
            text (str): Texto a procesar
            
        Returns:
            str: Texto procesado
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text 