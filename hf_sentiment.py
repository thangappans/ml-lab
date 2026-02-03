from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
test_sentences = [
    "Win a free mobile now",
    "I am very happy with this service",
"This is the worst experience ever",
"The product is okay",
"I am disappointed with the quality"]
for text in test_sentences:
    result = classifier(text)
    print("Text:", text)
    print("Prediction:", result)
    print()