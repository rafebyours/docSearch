import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker

# Download necessary resources from NLTK
nltk.download('stopwords')

# Initialize the stemmer and spell checker
stemmer = PorterStemmer()
spell = SpellChecker()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Split text into words
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    # Correct spelling
    words = [spell.correction(word) for word in words]
    
    # Remove None values (in case of spelling correction issues)
    words = [word for word in words if word is not None]
    
    # Join the words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text
