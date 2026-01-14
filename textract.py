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
    """Check if the file should be processed (i.e., no corresponding txt file exists)."""
    txt_path = os.path.splitext(os.path.abspath(file_path))[0] + '.txt'
    return not os.path.exists(txt_path)

def extract_text_from_image(image_path, client):
    """Extract text from the image using Amazon Textract."""
    with open(image_path, 'rb') as image:
        response = client.detect_document_text(Document={'Bytes': image.read()})
    
    extracted_text = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
    return '\n'.join(extracted_text)

def save_text_to_file(text, file_path):
    """Save the extracted text to a file."""
    txt_path = os.path.splitext(os.path.abspath(file_path))[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

def process_images_in_directory(directory):
    """Process all images in the given directory."""
    image_files = get_image_files(directory)
    textract_client = boto3.client('textract')  # create client once

    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            if should_process_file(image_path):
                try:
                    extracted_text = extract_text_from_image(image_path, textract_client)
                    save_text_to_file(extracted_text, image_path)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
            pbar.update(1)

# Usage in Jupyter notebook or standalone script
directory = '.'  # Current directory
process_images_in_directory(directory)

