# Amazon Review Analysis

Este proyecto analiza reseñas de Amazon utilizando técnicas de NLP y muestra los resultados en un dashboard interactivo.

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Unix o MacOS:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar recursos NLTK necesarios
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Estructura del Proyecto
```
amazon-review-analysis/
├── data/                  # Datos de entrada y procesados
├── models/               # Modelos entrenados
├── src/                  # Código fuente principal
├── app/                  # Aplicación Streamlit
└── requirements.txt     # Dependencias del proyecto
```

## Uso

1. Coloca los archivos TSV de Amazon en la carpeta `data/`
2. Ejecuta el procesamiento de datos:
   ```bash
   python src/data_processing.py
   ```
3. Entrena el modelo:
   ```bash
   python src/sentiment_classifier.py
   ```
4. Inicia la aplicación Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Licencia
MIT