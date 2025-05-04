import os
import shutil
import random
from pathlib import Path

def organize_dataset():
    # Create necessary directories
    base_dir = Path("weapon-dataset")
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    
    # Create train and val directories for both images and labels
    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    classes = ["drill", "knife"]
    train_ratio = 0.8  # 80% for training, 20% for validation
    
    for class_idx, class_name in enumerate(classes):
        source_dir = base_dir / class_name
        if not source_dir.exists():
            print(f"Warning: {source_dir} does not exist!")
            continue
            
        # Get all images (including .jpeg files)
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(list(source_dir.glob(ext)))
        
        if not images:
            print(f"No images found in {source_dir}")
            continue
            
        print(f"Found {len(images)} images in {class_name} directory")
        random.shuffle(images)
        
        # Split into train and val
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"Processing {len(train_images)} training images and {len(val_images)} validation images for {class_name}")
        
        # Copy images and create empty label files
        for img_path in train_images:
            # Copy image
            dest_path = images_dir / "train" / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path.name} to train directory")
            
            # Create empty label file
            label_path = labels_dir / "train" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0")  # Default bounding box (center of image)
        
        for img_path in val_images:
            # Copy image
            dest_path = images_dir / "val" / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path.name} to val directory")
            
            # Create empty label file
            label_path = labels_dir / "val" / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0")  # Default bounding box (center of image)
    
    print("Dataset organization complete!")

if __name__ == "__main__":
    organize_dataset() 