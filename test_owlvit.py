

import requests
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

print("Loading processor and model...")
# Load the processor and model from Hugging Face
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
print("Processor and model loaded.")

# Check if CUDA is available and move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# --- Image Setup ---
# URL of a sample image (you can replace this with a local file path if you prefer)
# Example local path: image_path = "your_image.jpg"
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
print(f"Loading image from: {image_url}")
try:
    # Load image using PIL (Python Imaging Library)
    image = Image.open(requests.get(image_url, stream=True).raw)
    print("Image loaded successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# --- Text Prompts ---
# Define what objects you want to detect
texts = [["a photo of a cat", "a photo of a remote control"]]
print(f"Text prompts: {texts}")

# --- Preprocess Inputs ---
print("Preprocessing inputs...")
inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

# --- Perform Object Detection ---
print("Running inference...")
with torch.no_grad(): # Ensure gradients are not calculated during inference
    outputs = model(**inputs)
print("Inference complete.")

# --- Process and Print Results ---
print("Processing results...")
# Target image sizes (used for scaling the bounding boxes)
target_sizes = torch.Tensor([image.size[::-1]]).to(device) # size is (width, height), need (height, width)

# Post-process the outputs (results are bounding boxes and scores)
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

i = 0 # Retrieve predictions for the first image
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

print("\n--- Detection Results ---")
for box, score, label in zip(boxes, scores, labels):
    box = [round(coord, 2) for coord in box.tolist()]
    print(f"Detected {text[label]}: Confidence {round(score.item(), 3)} at location {box}")

print("\nScript finished.")