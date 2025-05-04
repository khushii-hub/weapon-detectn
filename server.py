from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from weapon_detection import detect_weapons
import base64
import cv2
import numpy as np

app = Flask(__name__)
# Enable CORS for all routes to allow requests from your web application
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the image data from the request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'detected': False,
                'message': 'No image data provided'
            }), 400
        
        # Process the image with our weapon detection function
        image_data = data['image']
        # Convert base64 image to OpenCV format
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect weapons in the image
        result = detect_weapons(img)
        
        # Return the detection results
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'detected': False,
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Flask server for weapon detection...")
    print("Web interface will be available at: http://127.0.0.1:5001")
    print("API endpoint will be available at: http://127.0.0.1:5001/detect")
    app.run(debug=True, port=5001) 