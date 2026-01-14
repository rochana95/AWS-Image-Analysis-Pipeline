import os
import boto3
import json
import base64
from tqdm import tqdm
from PIL import Image

def get_image_files(directory):
    """Get all jpg and png files in the given directory."""
    return [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def should_process_file(file_path):
    """Check if the file should be processed (i.e., no corresponding txt file exists)."""
    txt_path = os.path.splitext(file_path)[0] + '.txt'
    return not os.path.exists(txt_path)

def analyze_image_with_bedrock(image_path):
    """Analyze the image using Amazon Bedrock."""
    bedrock_client = boto3.client('bedrock-runtime')

    # Convert the image to base64
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode()

    # Prepare the payload according to the Bedrock API requirements
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "Explain the content of this image."
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "anthropic_version": "bedrock-2023-05-31"
    }
    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',  # Replace with your preferred model ID
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )

        response_body = response['body'].read().decode('utf-8')  

        response_json = json.loads(response_body)

        # Adjust this based on the actual structure of the response
        analysis = response_json.get('message', {}).get('content', 'No analysis generated.')

        # Fallback to the full response if 'content' is missing
        if analysis == 'No analysis generated.':
            analysis = response_body

        return analysis

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error occurred during analysis."

def save_analysis_to_file(analysis, file_path):
    """Save the analysis to a file with a '_summary' suffix."""
    analysis_path = os.path.splitext(file_path)[0] + '_summary.txt'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis)

def process_images_in_directory(directory):
    """Process all images in the given directory."""
    image_files = get_image_files(directory)

    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for image_file in image_files:
            image_path = os.path.join(directory, image_file)

            if should_process_file(image_path):
                pbar.set_postfix({'Current file': image_file})
                analysis = analyze_image_with_bedrock(image_path)
                save_analysis_to_file(analysis, image_path)

            pbar.update(1)

# Usage in Jupyter notebook or standalone script
directory = '.'  # Current directory
process_images_in_directory(directory)    
