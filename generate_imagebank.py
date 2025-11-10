"""
Imagebank Generation Script for SEEG RSVP Experiments

This script processes all non-target images from each dataset and saves them
with crop information to imagebank/DATASET_NAME/framedata.json

USAGE:
------
python generate_imagebank.py
"""

import os
import random
import cv2
import numpy as np
import json
import screen_setup
from typing import Dict, List
from tqdm import tqdm
from image_datasets.datasets import Dataset, ILSVRC2012Dataset, OASISDataset, FERDataset, TrailerFacesHQDataset


### CONFIGURATION ###

class ImageBankConfig:
    """Configuration for imagebank generation"""
    def __init__(self):
        # Random seed for reproducibility
        self.RANDOM_SEED = 42


### IMAGEBANK GENERATOR ###

class ImageBankGenerator:
    """Main class for generating imagebank from datasets"""
    
    def __init__(self, config: ImageBankConfig, dataset: Dataset):
        self.config = config
        self.dataset = dataset
        self.framedata: List[Dict] = []
        
        # Initialize random seed
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
    
    def process_all_images(self, max_n_images: int = None) -> None:
        """Process all non-target images from the dataset"""
        non_target_images = self.dataset.get_non_target_images()
        dataset_name = self.dataset.get_dataset_name()
        
        print(f"\nProcessing non-target images from {dataset_name}...")

        if max_n_images is None:
            max_n_images = len(non_target_images)
        
        for image_filename in tqdm(non_target_images[:max_n_images], desc=f"{dataset_name}"):
            image_path = self.dataset.get_image_path(image_filename)
            
            # Generate random crop
            img = cv2.imread(image_path)
            if img is None:
                tqdm.write(f"Warning: Could not read image {image_path}, skipping...")
                continue
                
            top, left, size, _ = screen_setup.generate_random_crop(img)
            
            frame = {
                "dataset": dataset_name,
                "image_filename": image_filename,
                "image_path": image_path,
                "target": False,
                "crop": {
                    "top": top,
                    "left": left,
                    "width": size,
                    "height": size,
                }
            }
            self.framedata.append(frame)
    
    def save_imagebank(self, output_dir: str) -> None:
        """Save the imagebank to disk"""
        dataset_name = self.dataset.get_dataset_name()
        save_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Build imagebank data
        imagebank = {
            "dataset": dataset_name,
            "random_seed": self.config.RANDOM_SEED,
            "framedata": self.framedata,
            "total_frames": len(self.framedata)
        }
        
        # Save to file
        output_path = os.path.join(save_dir, "framedata.json")
        with open(output_path, "w") as f:
            json.dump(imagebank, f, indent=4)
        
        print(f"\nSaved {len(self.framedata)} frames to {output_path}")


### MAIN EXECUTION ###

def main():
    """Main execution function"""
    
    # Create configuration
    config = ImageBankConfig()
    
    print("\n=== Generating Imagebank ===")
    print(f"Random seed: {config.RANDOM_SEED}")
    print()
    
    # Output directory
    output_dir = "imagebank"
    
    # Initialize and process datasets
    datasets = []
    
    # ILSVRC2012 (ImageNet)
    print("\n--- Processing ILSVRC2012 (ImageNet) ---")
    ilsvrc_dataset = ILSVRC2012Dataset(
        base_path="image_datasets/ILSVRC2012_img_val/ILSVRC2012_img_val/",
        val_prep_path="image_datasets/ILSVRC2012_img_val/val_prep.sh",
        synset_words_path="image_datasets/ILSVRC2012_img_val/synset_words.txt",
        target_synset='dog.n.01',
        blacklisted_classes=[]
    )
    ilsvrc_dataset.preprocess()
    # datasets.append(ilsvrc_dataset)
    
    # OASIS
    print("\n--- Processing OASIS ---")
    oasis_dataset = OASISDataset(
        base_path="image_datasets/OASIS/OASIS",
        csv_path="image_datasets/OASIS/OASIS_bygender_CORRECTED_092617.csv",
        target_prefix="dog",
        blacklist_prefixes=["nude"]
    )
    oasis_dataset.preprocess()
    datasets.append(oasis_dataset)
    
    # Trailer Faces HQ
    trailer_faces_hq_dataset = TrailerFacesHQDataset(
        base_path="image_datasets/tfhqv2"
    )
    trailer_faces_hq_dataset.preprocess()
    datasets.append(trailer_faces_hq_dataset)
    
    # Process each dataset
    for dataset in datasets:
        generator = ImageBankGenerator(config, dataset)
        generator.process_all_images()#max_n_images=int(44100 / 0.7 * 0.15)) # 44100 is the number of images in the ILSVRC2012 dataset which are not target images
        generator.save_imagebank(output_dir)
    
    print("\n=== Done! ===")
    print(f"All imagebanks saved to {output_dir}/")


if __name__ == "__main__":
    main()
