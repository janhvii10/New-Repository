# Import required libraries
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample Product Reviews Dataset
data = {
    "Review": [
        "The product is very good and works perfectly",
        "I am extremely disappointed with the quality",
        "This item was delivered quickly and works well",
        "Worst product ever, not recommended at all"
    ]
}

df = pd.DataFrame(data)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Apply preprocessing
df['Tokens'] = df['Review'].apply(preprocess_text)

# Apply Stemming
df['Stemming'] = df['Tokens'].apply(
    lambda x: [stemmer.stem(word) for word in x]
)

# Apply Lemmatization
df['Lemmatization'] = df['Tokens'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x]
)

# Display results
print("Original Reviews:\n", df['Review'], "\n")
print("After Stemming:\n", df['Stemming'], "\n")
print("After Lemmatization:\n", df['Lemmatization'])