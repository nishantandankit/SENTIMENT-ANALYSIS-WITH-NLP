import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

reviews_data = {
    'review': [
        # Positive Reviews
        'This product is amazing! I love it.',
        'It works perfectly, just as described.',
        'I am very happy with my purchase.',
        'Highly recommended to everyone.',
        'Great quality and fast shipping.',
        'A wonderful experience, will buy again.',
        'Absolutely fantastic, five stars!',
        'The best I have ever used.',
        # Negative Reviews
        'Absolutely terrible, do not buy this.',
        'A complete waste of money.',
        'The quality is poor and it broke.',
        'I am so disappointed with this item.',
        'Did not work at all, very frustrating.',
        'This was a huge mistake.',
        'Awful product, terrible service.',
        'Broke after one use, not worth it.'
    ],
    'sentiment': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 8 positive, 8 negative
}
df = pd.DataFrame(reviews_data)

print("--- Sample Customer Reviews ---")
print(df.head()) 

X_text = df['review']
y = df['sentiment']

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X_text)

# --- Split Data and Train the Model (with Stratification) ---
# Total 16 samples. The split will be 12 for training and 4 for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.25, random_state=42, stratify=y
)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

print("\n--- Logistic Regression Model Training Complete ---")

y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Test Set Predictions: {y_pred}")
print(f"Actual Sentiments:    {y_test.values}")
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)
