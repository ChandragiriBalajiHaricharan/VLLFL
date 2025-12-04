import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import Owlv2Processor

class AgricultureObjectDetectionDataset(Dataset):
    """
    PyTorch Dataset for loading images and LabelMe JSON annotations
    from a single directory. Assumes JPG/PNG images and .json annotation files
    with the same base name reside in the same directory.
    """
    def __init__(self, data_dir, processor, texts_to_find, fruit_type="unknown"):
        """
        Initializes the dataset.

        Args:
            data_dir (str): Path to the directory containing JPG images and JSON annotations.
            processor (Owlv2Processor): The loaded OWL-ViT processor.
            texts_to_find (list): A list of text descriptions for objects.
            fruit_type (str): Name of the fruit/class this dataset instance represents.
        """
        self.data_dir = data_dir
        self.processor = processor
        self.texts_to_find = texts_to_find
        self.fruit_type = fruit_type

        # Find all image files in the data directory
        try:
            image_extensions = ('.jpg', '.jpeg', '.png')
            all_files = os.listdir(data_dir)
            self.image_files = sorted([f for f in all_files
                                if f.lower().endswith(image_extensions)])

            # Keep only images that have a corresponding JSON file
            self.valid_files = []
            for img_name in self.image_files:
                base_name = os.path.splitext(img_name)[0]
                ann_path = os.path.join(self.data_dir, base_name + ".json")
                if os.path.exists(ann_path):
                    self.valid_files.append(img_name)

            self.image_files = self.valid_files # Update list to only include valid pairs

            if not self.image_files:
                 print(f"Warning: No image files with corresponding JSON annotations found in {data_dir}")
            else:
                 print(f"Found {len(self.image_files)} image/JSON pairs in {data_dir} for fruit type '{self.fruit_type}'")

        except FileNotFoundError:
            print(f"ERROR: Data directory not found at {data_dir}")
            self.image_files = []
        except Exception as e:
            print(f"Error reading data directory {data_dir}: {e}")
            self.image_files = []

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Loads and preprocesses one sample from the dataset."""
        if idx >= len(self.image_files):
             raise IndexError("Index out of bounds")

        # --- Load Image ---
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        try:
            # Ensure image is in RGB format
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None # Handle error appropriately

        # --- Load Annotations (Using LabelMe format structure) ---
        base_name = os.path.splitext(img_name)[0]
        # Adjust extension if your JSON files have a different naming convention
        ann_path = os.path.join(self.data_dir, base_name + ".json")
        ground_truth_boxes = []
        ground_truth_labels = []
        try:
            with open(ann_path, 'r') as f:
                annotations = json.load(f)

            # --- Extract bounding boxes and labels from LabelMe format ---
            if "shapes" in annotations and isinstance(annotations["shapes"], list):
                for shape in annotations["shapes"]:
                    if (shape.get("shape_type") == "rectangle" and
                            shape.get("label") and # Ensure label exists
                            isinstance(shape.get("points"), list) and
                            len(shape["points"]) == 2):

                        label = shape["label"]
                        points = shape["points"]
                        point1 = points[0]
                        point2 = points[1]

                        if (len(point1) == 2 and len(point2) == 2 and
                            all(isinstance(coord, (int, float)) for coord in point1 + point2)):

                            # Calculate [xmin, ymin, xmax, ymax]
                            xmin = min(point1[0], point2[0])
                            ymin = min(point1[1], point2[1])
                            xmax = max(point1[0], point2[0])
                            ymax = max(point1[1], point2[1])

                            if xmax > xmin and ymax > ymin:
                                bbox = [xmin, ymin, xmax, ymax]
                                ground_truth_boxes.append(bbox)
                                ground_truth_labels.append(label)
                            else:
                                print(f"Warning: Degenerate box found in {ann_path}")
                        else:
                            print(f"Warning: Invalid points format in shape in file {ann_path}")

        except Exception as e:
            print(f"Error loading/parsing annotation {ann_path}: {e}")
            return None

        # --- Preprocess using Owlv2Processor (FIXED WITH TRUNCATION) ---
        try:
            # For OWL-ViT: if texts_to_find is [[list]], extract single list
            # The processor expects either a single string or a list of strings
            if isinstance(self.texts_to_find, list) and len(self.texts_to_find) > 0:
                if isinstance(self.texts_to_find[0], list):
                    flat_texts = self.texts_to_find[0]
                else:
                    flat_texts = self.texts_to_find
            else:
                flat_texts = self.texts_to_find
            
            # Process with proper tensor handling
            inputs = self.processor(
                text=[flat_texts],     # Wrap in list for batch processing
                images=image, 
                return_tensors="pt"
            )
            
            # CRITICAL FIX: Truncate input_ids to max_position_embeddings (16)
            # This matches the model's position embedding size
            max_length = 16
            inputs['input_ids'] = inputs['input_ids'][:, :max_length]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_length]
            # Squeeze the batch dimension added by the processor
            inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        except Exception as e:
             print(f"Error during Owlv2Processor processing for image {img_path}: {e}")
             return None

        # --- Prepare Targets ---
        targets = {
            "boxes": torch.tensor(ground_truth_boxes, dtype=torch.float32) if ground_truth_boxes else torch.empty((0, 4), dtype=torch.float32),
            "labels": ground_truth_labels 
        }

        # Add image_id and original size
        targets["image_id"] = torch.tensor([idx])
        targets["orig_size"] = torch.tensor([image.height, image.width])

        return inputs, targets

# --- Example Usage ---
if __name__ == '__main__':
    METAFRUIT_ROOT = r"C:\Users\balaj\OneDrive\Documents\project sem 5\archive\MetaFruit"
    FRUIT_FOLDERS = ["apple", "grapefruit", "lemon", "orange", "tangerine"]

    if not os.path.isdir(METAFRUIT_ROOT):
        print(f"ERROR: MetaFruit root directory not found at {METAFRUIT_ROOT}")
    else:
        print("Loading Owlv2Processor...")
        try:
            processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            print("Processor loaded.")
        except Exception as e:
            print(f"Error loading Owlv2Processor: {e}")
            exit()

        query_texts = [[ "apple", "grapefruit", "lemon", "orange", "tangerine", "leaf", "person" ]] 
        
        all_datasets = []
        print("\nCreating dataset instances...")
        for fruit in FRUIT_FOLDERS:
            data_dir = os.path.join(METAFRUIT_ROOT, fruit)
            if os.path.isdir(data_dir):
                print(f"---> Processing folder: {data_dir}")
                dataset_instance = AgricultureObjectDetectionDataset(
                    data_dir=data_dir,
                    processor=processor,
                    texts_to_find=query_texts,
                    fruit_type=fruit
                )
                if len(dataset_instance) > 0:
                    all_datasets.append(dataset_instance)

        if all_datasets:
            combined_dataset = ConcatDataset(all_datasets)
            print(f"\nSuccessfully created combined dataset with {len(combined_dataset)} total samples.")
            
            # Test loading sample 0
            print("\nAttempting to load sample 0...")
            result = combined_dataset[0]
            if result:
                print("Sample 0 loaded successfully!")
                print("Input IDs shape:", result[0]['input_ids'].shape) # Should be [1, 16] or [16]
        else:
            print("\nERROR: No valid datasets created.")