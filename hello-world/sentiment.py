from transformers import pipeline

# Load a sentiment analysis pileline
sentiment_model = pipeline("sentiment-analysis")

while True:
    text = input("Enter a sentence (or ''quit' to exit): ")
    if text.lower() == "quit":
        break
    result = sentiment_model(text)[0]
    print(f"Label: {result['label']}, Confidence: {result['score']:.2f}")
