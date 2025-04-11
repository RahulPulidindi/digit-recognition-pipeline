import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import time

# Force CPU usage and optimize
device = torch.device('cpu')
torch.set_num_threads(4)  # Optimize for a 4-core CPU
print(f"Using device: {device} with {torch.get_num_threads()} threads")

# Define the same lightweight model architecture as in training
class LightweightNet(nn.Module):
    def __init__(self):
        super(LightweightNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Initialize Flask app
app = Flask(__name__)

# Load model
model = None

def load_model():
    global model
    model = LightweightNet().to(device)
    model_path = os.path.join('/models', 'mnist_model.pt')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")
        
        # JIT compile the model for faster CPU inference
        example = torch.rand(1, 1, 28, 28).to(device)
        model = torch.jit.trace(model, example)
        print("Model JIT compiled for faster inference")
    else:
        print(f"Error: Model not found at {model_path}")

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Get image data from request
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale and invert (so digits are black on white background)
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    
    # Transform image
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Measure inference time
    start_time = time.time()
    
    # Make prediction
    with torch.no_grad():
        # Run inference 3 times and take the average (first run often has JIT overhead)
        for _ in range(2):  # Warmup runs
            _ = model(tensor)
            
        # Actual timed run
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        prediction = predicted.item()
    
    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000  # convert to milliseconds
    
    return jsonify({
        'prediction': str(prediction),
        'inference_time_ms': f"{inference_time:.2f}"
    })

if __name__ == '__main__':
    # Load model when app starts
    load_model()
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)