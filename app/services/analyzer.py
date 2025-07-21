from langdetect import detect
from transformers import pipeline

# Load the English sentiment analysis pipeline
en_sentiment = pipeline("sentiment-analysis")

# Load the Korean sentiment analysis pipeline using a pretrained model
ko_sentiment = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

def analyze_text(text: str):
    """
    Analyze the sentiment of the input text.
    Automatically detects language (English or Korean).
    Returns error if the input is too short or language is unsupported.
    """
    # Check if the input is empty or too short
    if not text or len(text.strip()) < 5:
        return {"error": "The input text is too short. Please enter at least 5 characters."}
    
    # Detect the language ('en' for English, 'ko' for Korean, etc.)
    lang = detect(text)
    
    # Only support Korean and English
    if lang not in ["ko", "en"]:
        return {"error": f"Unsupported language: {lang}"}
    
    # Analyze with the corresponding sentiment model
    if lang == "ko":
        result = ko_sentiment(text)
        return {"lang": "ko", "result": result}
    else:
        result = en_sentiment(text)
        return {"lang": "en", "result": result}
