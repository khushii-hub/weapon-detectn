import requests
import base64
import json
import cv2
import numpy as np

def test_detection(image_path):
    # Read the image
    with open(image_path, 'rb') as image_file:
        # Convert image to base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        base64_image = f"data:image/jpeg;base64,{base64_image}"
    
    # Prepare the request
    url = "http://127.0.0.1:5001/detect"
    headers = {'Content-Type': 'application/json'}
    data = {'image': base64_image}
    
    # Send the request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Print the results
    print(f"\nTesting image: {image_path}")
    print("Response:", response.json())

if __name__ == "__main__":
    # Test with images from your dataset
    test_images = [
        "weapon-dataset/drill/drill_0.jpeg",
        "weapon-dataset/knife/knife_0.jpeg"
    ]
    
    for image_path in test_images:
        test_detection(image_path) 