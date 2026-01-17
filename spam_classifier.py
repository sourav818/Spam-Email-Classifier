import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test accuracy
predictions = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Test with your own messages
while True:
    msg = input("\nEnter a message (or type exit): ")
    if msg.lower() == "exit":
        break

    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)[0]
    print("SPAM ❌" if result == 1 else "NOT SPAM ✅")
