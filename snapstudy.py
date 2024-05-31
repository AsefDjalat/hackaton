import pytesseract
from PIL import Image
import spacy
from transformers import pipeline
import subprocess
import sys

# Function to install SpaCy model
# def install_spacy_model():
#     try:
#         spacy.load("en_core_web_sm")
#     except OSError:
#         print("Downloading SpaCy 'en_core_web_sm' model...")
#         subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
#         print("Model downloaded successfully.")

# Step 1: Image Preprocessing and OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Step 2: Text Cleaning
def clean_text(text):
    # Basic cleaning: removing extra spaces, correcting OCR errors, etc.
    return ' '.join(text.split())

# Step 3: Summarization
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Step 4: Keyword Extraction
def extract_keywords(text):
    #install_spacy_model()  # Ensure the model is installed
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if chunk.text]
    return keywords

# Step 5: Flashcard Generation
def generate_flashcards(keywords, text):
    flashcards = []
    for keyword in keywords:
        question = f"What is {keyword}?"
        answer = f"{keyword} is {text.partition(keyword)[2][:100]}..."  # Basic example
        flashcards.append({'question': question, 'answer': answer})
    return flashcards

# Example Usage
image_path = 'test.png'
text = extract_text_from_image(image_path)
cleaned_text = clean_text(text)
summary = summarize_text(cleaned_text)
keywords = extract_keywords(text)
flashcards = generate_flashcards(keywords, cleaned_text)

print("Summary:", summary)
print("Keywords:", keywords)
print("Flashcards:", flashcards)
#