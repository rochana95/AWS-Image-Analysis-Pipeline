import os
import boto3
from tqdm import tqdm
from PIL import Image

def get_image_files(directory):
    """Get all jpg, jpeg, and png files in the given directory (skip folders)."""
    return [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

def should_process_file(file_path):
    """Check if the file should be processed (i.e., no corresponding txt and summary files exist)."""
    txt_path = os.path.splitext(os.path.abspath(file_path))[0] + '.txt'
    summary_path = os.path.splitext(os.path.abspath(file_path))[0] + '_summary.txt'
    return not (os.path.exists(txt_path) and os.path.exists(summary_path))

def extract_text_from_image(image_path, client):
    """Extract text from the image using Amazon Textract."""
    with open(image_path, 'rb') as image:
        response = client.detect_document_text(Document={'Bytes': image.read()})
    return '\n'.join([item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE'])

def summarize_text(text, client):
    """Summarize the extracted text using Amazon Comprehend."""
    if len(text) > 5000:  # Comprehend limit
        text = text[:5000]

    key_phrases_response = client.detect_key_phrases(Text=text, LanguageCode='en')
    key_phrases = [phrase['Text'] for phrase in key_phrases_response['KeyPhrases']]

    sentiment_response = client.detect_sentiment(Text=text, LanguageCode='en')
    sentiment = sentiment_response['Sentiment']

    summary = "Summary:\n" + '\n'.join(key_phrases[:5])  # Top 5 key phrases
    summary += f"\n\nSentiment: {sentiment}"
    return summary

def save_text_to_file(text, file_path):
    """Save the extracted text to a file."""
    txt_path = os.path.splitext(os.path.abspath(file_path))[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_summary_to_file(summary, file_path):
    """Save the summary to a file with a '_summary' suffix."""
    summary_path = os.path.splitext(os.path.abspath(file_path))[0] + '_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

def process_images_in_directory(directory):
    """Process all images in the given directory."""
    image_files = get_image_files(directory)
    textract_client = boto3.client('textract')
    comprehend_client = boto3.client('comprehend')

    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            if should_process_file(image_path):
                try:
                    extracted_text = extract_text_from_image(image_path, textract_client)
                    save_text_to_file(extracted_text, image_path)

                    summary = summarize_text(extracted_text, comprehend_client)
                    save_summary_to_file(summary, image_path)

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
            pbar.update(1)

# Usage in Jupyter notebook or standalone script
directory = '.'  # Current directory
process_images_in_directory(directory)
