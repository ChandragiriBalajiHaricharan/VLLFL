import os
import io
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from peft import PeftModel

app = Flask(__name__)
CORS(app)  # Important: Allows your Frontend to talk to the Backend

# --- CONFIGURATION ---
DEVICE = "cpu"
# If you have GPU setup and want to use it, change to "cuda"
# DEVICE = "cuda" 

CHECKPOINT = "google/owlv2-base-patch16-ensemble"
ADAPTER_PATH = "vllfl_adapters_efficient"

print(f"🚀 Initializing VLLFL Backend on {DEVICE}...")

# 1. Load Base Model (Guaranteed to work)
try:
    processor = Owlv2Processor.from_pretrained(CHECKPOINT)
    base_model = Owlv2ForObjectDetection.from_pretrained(CHECKPOINT)
    print("✅ Base Model Loaded (Google OWL-ViT)")
except Exception as e:
    print(f"❌ Critical Error: Could not download model.\n{e}")
    exit(1)

# 2. Try to Load Trained Adapters (Falls back to Base Model if missing)
if os.path.exists(ADAPTER_PATH) and os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.bin")):
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("✅ SUCCESS: Loaded Your Custom Fine-Tuned Adapters!")
    except:
        print("⚠️ Adapters found but failed to load. Using Base Model.")
        model = base_model
else:
    print("ℹ️  No training files found. Using BASE MODEL for Demo.")
    model = base_model

model.to(DEVICE)
model.eval()

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    # This prevents the 404 error when you open the link in browser
    return "<h1>✅ VLLFL Server is Running!</h1><p>Go to your index.html file to use the app.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read Image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Get Prompts (Default to fruits if empty)
        user_texts = request.form.get('texts', 'apple, orange, lemon, person')
        texts = [[t.strip() for t in user_texts.split(',')]]
        
        print(f"📸 Analyzing image for: {texts[0]}")

        # Inference
        inputs = processor(text=texts, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-Processing
        # 1. First Layer of Defense: Ask processor to give us > 0.7
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

        predictions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            
            # 2. Second Layer of Defense (MANUAL LOCK)
            # Strictly ignore anything less than 0.7 (70%)
            if score < 0.7:
                continue

            predictions.append({
                "label": texts[0][label],
                "score": round(float(score), 2),
                "box": [round(float(i), 2) for i in box.tolist()]
            })
        
        print(f"✅ Found {len(predictions)} objects (Only >70%).")
        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Runs on all network interfaces
    app.run(host='0.0.0.0', port=5000)