# --- 1. SETUP: IMPORT LIBRARIES AND DOWNLOAD DATA ---

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords from NLTK (only needs to be done once)
# Stopwords are common words like 'the', 'is', 'in' that are often removed in NLP.
nltk.download('stopwords')
print("--- Step 1: Libraries Imported and Setup Complete ---")


# --- 2. LOAD AND PREPROCESS THE DATASET ---

# Load the IMDb Movie Reviews dataset from a public URL
# This dataset contains 50,000 reviews, labeled as 'positive' or 'negative'.
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' # This is a common source
# For simplicity, we'll use a pre-processed CSV version available online.
csv_url = 'https://raw.githubusercontent.com/Ankit152/IMDB-Sentiment-Analysis/master/IMDB-Dataset.csv'

print("\n--- Step 2: Loading and Preprocessing Data ---")
print(f"Loading dataset from: {csv_url}")
df = pd.read_csv(csv_url)

# Let's look at the first few rows and the data balance
print("\nDataset Head:")
print(df.head())
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# Preprocessing Steps:
# 1. Map sentiment labels to numerical values (positive: 1, negative: 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 2. Clean the review text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Function to clean raw text data.
    - Removes HTML tags
    - Removes punctuation and special characters
    - Converts to lowercase
    - Removes stopwords
    """
    # Remove HTML tags using regex
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words and remove stopwords
    words = text.split()
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)

# Apply the cleaning function to all reviews
print("\nCleaning and preprocessing text reviews...")
df['cleaned_review'] = df['review'].apply(clean_text)
print("Text cleaning complete. Sample of cleaned data:")
print(df[['review', 'cleaned_review']].head())


# --- 3. SPLIT DATA AND PERFORM TF-IDF VECTORIZATION ---

print("\n--- Step 3: Splitting Data and Applying TF-IDF ---")
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'],
    df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment'] # Ensures the split maintains the same proportion of sentiments
)

# TF-IDF Vectorization: Converts text into a matrix of TF-IDF features.
# It gives more weight to words that are frequent in a document but rare across all documents.
vectorizer = TfidfVectorizer(max_features=5000) # Use the top 5000 most frequent words

# Fit the vectorizer on the training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transform the test data using the already fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)

print(f"Data vectorized. Training data shape: {X_train_tfidf.shape}")


# --- 4. TRAIN THE LOGISTIC REGRESSION MODEL ---

print("\n--- Step 4: Training the Logistic Regression Model ---")
# Logistic Regression is a simple yet powerful linear model for classification.
model = LogisticRegression(solver='liblinear', random_state=42)

# Train the model on the TF-IDF features of the training data
model.fit(X_train_tfidf, y_train)
print("Model training complete.")


# --- 5. EVALUATE THE MODEL ---

print("\n--- Step 5: Evaluating Model Performance ---")
# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Show a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# --- 6. SHOWCASING SENTIMENT EVALUATION (DELIVERABLE) ---

print("\n--- Step 6: Testing the Model on New Reviews ---")

def predict_sentiment(review_text):
    """
    Takes a new review, cleans it, vectorizes it, and predicts the sentiment.
    """
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    probability = model.predict_proba(vectorized)

    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    confidence = probability[0][prediction[0]]

    print(f"Review: '{review_text}'")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")

# Example reviews
predict_sentiment("This movie was absolutely brilliant! The acting was superb and the plot was gripping.")
predict_sentiment("A complete waste of time. The plot was predictable and the acting was terrible.")
predict_sentiment("It was an okay movie, not great but not bad either. I might watch it again.")
