import os
import json
import numpy as np
import cv2
import argparse

# Define the classes and the mapping from class name to index
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

def generate_annotations(image_dir, label_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get image and annotation filenames
    png_files = sorted([
        os.path.join(image_dir, fname) 
        for fname in os.listdir(image_dir) if fname.endswith('.png')
    ])
    
    json_files = sorted([
        os.path.join(label_dir, fname)
        for fname in os.listdir(label_dir) if fname.endswith('.json')
    ])

    assert len(png_files) == len(json_files), "Mismatch in images and annotations"

    for image_path, json_path in zip(png_files, json_files):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # Initialize the label array in (num_classes, height, width) shape
        label = np.zeros((len(CLASSES), height, width), dtype=np.uint8)

        with open(json_path, "r") as f:
            annotations = json.load(f)["annotations"]

        for ann in annotations:
            class_name = ann["label"]
            class_index = CLASS2IND[class_name]
            points = np.array(ann["points"], dtype=np.int32)

            # Create a mask for the current class
            class_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(class_mask, [points], 1)
            
            # Assign the mask to the appropriate class channel
            label[class_index] = class_mask

        # Save the label in (num_classes, height, width) shape
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.npy")
        np.save(output_path, label)
        print(f"Saved annotation for {image_path} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .npy annotations for training images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory of training images")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to the directory of JSON label files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save generated .npy files")

    args = parser.parse_args()

    generate_annotations(args.image_dir, args.label_dir, args.output_dir)
