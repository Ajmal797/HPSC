import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from transformers import pipeline
from joblib import Parallel, delayed

# Function to predict sentiment of a new statement
def predict_sentiment(new_statement, classifier, vectorizer):
    # Transform the new statement using the trained vectorizer
    new_statement_tfidf = vectorizer.transform([new_statement])
    # Predict the sentiment using the trained model
    prediction = classifier.predict(new_statement_tfidf)
    # Map the prediction back to 'positive' or 'negative'
    sentiment = 'positive' if prediction[0] == 1 else 'negative'
    return sentiment

# Emotion detection function (using BERT-based model)
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

def predict_emotion_bert(text):
    result = emotion_classifier(text)
    return result

# Load the datasets
education_df = pd.read_csv('/home/student1/Education.csv')  # Replace with your education dataset path
sports_df = pd.read_csv('/home/student1/Sports.csv')  # Replace with your sports dataset path

# Combine the datasets into one
combined_df = pd.concat([education_df, sports_df], ignore_index=True)

# Preprocessing text
X = combined_df['Text']
y = combined_df['Label'].map({'positive': 1, 'negative': 0})  # Encoding labels (for sentiment)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train models for sentiment prediction
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Fit models
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)

# Function to calculate accuracy of all models
def get_accuracy(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Function for parallel processing of sentiment analysis
def parallel_predict(models, statement, vectorizer):
    results = Parallel(n_jobs=-1)(delayed(predict_sentiment)(statement, model, vectorizer) for model in models.values())
    return results

# Interactive interface
print("Sentiment and Emotion Analysis Tool\n")
statement_input = input("Enter a statement to analyze: ").strip()

if statement_input:
    print("\nProcessing...")

    # Non-parallel sentiment analysis
    start_time = time.time()
    non_parallel_results = {}
    non_parallel_accuracy = {}
    for model_name, model in models.items():
        sentiment = predict_sentiment(statement_input, model, vectorizer)
        non_parallel_results[model_name] = sentiment
        accuracy = get_accuracy(model, X_test_tfidf, y_test)
        non_parallel_accuracy[model_name] = accuracy
    non_parallel_time = time.time() - start_time

    # Display non-parallel results
    print("\nNon-Parallel Sentiment Analysis Results:")
    for model_name, sentiment in non_parallel_results.items():
        sentiment_color = "Positive" if sentiment == "positive" else "Negative"
        print(f"Model: {model_name} => Sentiment: {sentiment_color} (Accuracy: {non_parallel_accuracy[model_name] * 100:.2f}%)")

    # Parallel sentiment analysis
    start_time = time.time()
    parallel_results = parallel_predict(models, statement_input, vectorizer)
    parallel_time = time.time() - start_time

    # Display parallel results
    print("\nParallel Sentiment Analysis Results:")
    for model_name, sentiment in zip(models.keys(), parallel_results):
        sentiment_color = "Positive" if sentiment == "positive" else "Negative"
        accuracy = get_accuracy(models[model_name], X_test_tfidf, y_test)
        print(f"Model: {model_name} => Sentiment: {sentiment_color} (Accuracy: {accuracy * 100:.2f}%)")

    # Speedup and Efficiency Calculation
    speedup = non_parallel_time / parallel_time
    efficiency = speedup / 4  # Assuming 4 cores used in parallel processing
    print(f"\nExecution Time (Non-parallel): {non_parallel_time:.2f} seconds")
    print(f"Execution Time (Parallel): {parallel_time:.2f} seconds")
    print(f"Speedup: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2f}")

    # Emotion Analysis
    emotions = predict_emotion_bert(statement_input)
    print("\nEmotion Analysis Result:")
    print(emotions)
else:
    print("Please enter a valid statement to analyze.")
