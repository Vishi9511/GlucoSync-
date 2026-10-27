
from transformers import pipeline
import sys

print("Loading the Food-101 model...")
print("The first run may take a few minutes.")

classifier = pipeline(
    "image-classification",
    model="nateraw/food"
)

image_path = sys.argv[1]

print("\nAnalyzing your food photo...")
results = classifier(image_path, top_k=5)

print("\nTop predictions:")
for result in results:
    label = result["label"].replace("_", " ").title()
    confidence = result["score"] * 100
    print(f"{label}: {confidence:.1f}%")
