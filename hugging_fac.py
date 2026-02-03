# Step 1: Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Step 2: Dataset
messages = [
    "Win a free mobile now",
    "Limited time offer",
    "Claim your prize",
    "Congratulations you won",
    "Free recharge available",
    "How are you",
    "Let's meet tomorrow",
    "Call me when free",
    "Are you coming today",
    "See you at office"
]
labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Ham", "Ham", "Ham", "Ham", "Ham"
]
# Step 3: Convert text to numbers (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)
# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)
# Step 5: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)
# Step 6: Test model
predictions = model.predict(X_test)
# Step 7: Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))
# Step 8: Test new message
new_message = ["Free prize offer"]
new_vector = vectorizer.transform(new_message)
result = model.predict(new_vector)
print("Message:", new_message[0])
print("Prediction:", result[0])