## Libraries

# language detection
from langdetect import detect

## Language Detection

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'  
    return lang
