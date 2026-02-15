"""
Template Generation Script for SEEG RSVP Experiments

This script generates experiment templates with target and non-target frames.

USAGE:
------
1. Standard mode (generate all frames from scratch):
   python template_generate.py

2. Reuse mode (use existing non-target frames, generate new targets):
   python template_generate.py <source_json_path> <template_prefix>
   
   Example:
   python template_generate.py templates/first_n480/template.json second
   
   This mode:
   - Loads non-target frames from the existing template JSON
   - Generates fresh target frames based on config constraints
   - Combines and shuffles all frames
   - Copies images to new template directory
   - Saves the new template with the provided prefix and updated timing/seed
   
   Note: template_prefix is REQUIRED when providing a source JSON

PROGRAMMATIC USAGE:
------------------
from template_generate import main

# Standard mode
main()

# Reuse mode (template_prefix is required)
main(source_framedata_json="path/to/existing/template.json", 
     template_prefix="second")
"""

import os
import random
import cv2
import numpy as np
import json
import shutil
import time
import screen_setup
import math
from tqdm import tqdm
from typing import Dict, List, Optional
from image_datasets.datasets import Dataset, ILSVRC2012Dataset, OASISDataset, FERDataset, TrailerFacesHQDataset


### CONFIGURATION ###

class TemplateConfig:
    """Configuration for template generation"""
    def __init__(self, template_prefix: Optional[str] = None):
        # Frame settings
        self.N_FRAMES = 960
        self.N_REPEATS = 2
        # Random seed for reproducibility and session variation
        self.RANDOM_SEED_STRING = "3"
        
        # Timing settings (in ms)
        self.TIME_ON_FROM = 100
        self.TIME_ON_TO = 100
        self.TIME_OFF_FROM = 150 - 25
        self.TIME_OFF_TO = 150 + 25

        # Flipped timing settings (as before before the timing was changed)
        # self.TIME_ON_FROM = 150 - 25
        # self.TIME_ON_TO = 150 + 25
        # self.TIME_OFF_FROM = 100
        # self.TIME_OFF_TO = 100

        # # for the first tutorial session, we will use the following settings
        # self.N_FRAMES = 160
        # self.N_REPEATS = 2
        # self.TIME_ON_FROM = 100
        # self.TIME_ON_TO = 100
        # self.TIME_OFF_FROM = 400 - 25
        # self.TIME_OFF_TO = 400 + 25
        # self.RANDOM_SEED_STRING = "T"
        
        # Derived timing
        self.TIME_ON_MEAN = (self.TIME_ON_FROM + self.TIME_ON_TO) / 2
        self.TIME_OFF_MEAN = (self.TIME_OFF_FROM + self.TIME_OFF_TO) / 2
        
        # Distance constraints
        self.MIN_DISTANCE_BETWEEN_TARGET_FRAMES = math.ceil(
            2 * 1000 / (self.TIME_ON_MEAN + self.TIME_OFF_MEAN)
        )
        self.MAX_DISTANCE_BETWEEN_TARGET_FRAMES = math.ceil(
            20 * 1000 / (self.TIME_ON_MEAN + self.TIME_OFF_MEAN)
        )
        self.N_TARGET_FRAMES = math.ceil(
            self.N_FRAMES * 2 / (self.MAX_DISTANCE_BETWEEN_TARGET_FRAMES + self.MIN_DISTANCE_BETWEEN_TARGET_FRAMES)
        )
        
        self.TEMPLATE_PREFIX = template_prefix if template_prefix is not None else "trial"
        
        # Derived properties
        self.TEMPLATE_NAME = (
            f"{self.TEMPLATE_PREFIX}_"
            f"s{self.RANDOM_SEED_STRING}_"
            f"n{self.N_FRAMES}_x{self.N_REPEATS}_"
            f"on{self.TIME_ON_FROM}-{self.TIME_ON_TO}_"
            f"off{self.TIME_OFF_FROM}-{self.TIME_OFF_TO}"
        )
        self.RANDOM_SEED = (
            int.from_bytes((self.RANDOM_SEED_STRING + "salt_for_reproducibility").encode(), 'little') * 19241
        ) % (2**32)
        
        # Dataset selection distribution
        self.DATASET_DISTRIBUTION = {
            "ILSVRC2012_img_val": 0.7,
             "OASIS": 0.15 if not self.TEMPLATE_PREFIX == "tutorial" else 0.0, # Remove OASIS for the tutorial session
             "TrailerFacesHQ": 0.15,
            # "FER": 0.0  # remove FER for now
        }
    
    def pick_dataset(self) -> str:
        """Randomly pick a dataset based on distribution"""
        r = random.random()
        cumulative = 0.0
        
        for dataset_name, prob in self.DATASET_DISTRIBUTION.items():
            cumulative += prob
            if r < cumulative:
                return dataset_name
        
        # Fallback to first dataset if something goes wrong
        return list(self.DATASET_DISTRIBUTION.keys())[0]


### TEMPLATE GENERATOR ###

class TemplateGenerator:
    """Main class for generating experiment templates"""
    
    def __init__(self, config: TemplateConfig, datasets: Dict[str, Dataset], source_framedata_json: Optional[str] = None):
        self.config = config
        self.datasets = datasets
        self.already_picked_filenames: set = set()
        self.framedata: List[Dict] = []
        self.source_framedata_json = source_framedata_json
        
        # Initialize random seeds
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # Calculate predetermined timing
        self._calculate_timing()
        
    def _calculate_timing(self) -> None:
        """Pre-calculate stimulus onset and offset times"""
        n_stimuli = self.config.N_FRAMES * self.config.N_REPEATS
        self.stimulus_onset_times = np.zeros(n_stimuli)
        self.stimulus_offset_times = np.zeros(n_stimuli)
        
        time_on_mean = self.config.TIME_ON_MEAN
        time_off_mean = self.config.TIME_OFF_MEAN

        # # #Old, mistaken timing calculation
        # self.stimulus_offset_times[0] = (time_on_mean + time_off_mean) / 1000
        # for i in range(1, n_stimuli):
        #     on_time = random.uniform(self.config.TIME_ON_FROM, self.config.TIME_ON_TO) / 1000
        #     off_time = random.uniform(self.config.TIME_OFF_FROM, self.config.TIME_OFF_TO) / 1000
            
        #     self.stimulus_onset_times[i] = self.stimulus_offset_times[i-1] + on_time
        #     self.stimulus_offset_times[i] = self.stimulus_onset_times[i] + off_time
        
        # # # New, correct timing calculation
        self.stimulus_onset_times[0] = (time_on_mean+time_off_mean) / 1000 # Start one stimulus cycle later to allow the task laptop some time for processing
        self.stimulus_offset_times[0] = self.stimulus_onset_times[0] + time_on_mean / 1000
        for i in range(1, n_stimuli):
            on_time = random.uniform(self.config.TIME_ON_FROM, self.config.TIME_ON_TO) / 1000
            off_time = random.uniform(self.config.TIME_OFF_FROM, self.config.TIME_OFF_TO) / 1000
            
            self.stimulus_onset_times[i] = self.stimulus_offset_times[i-1] + off_time
            self.stimulus_offset_times[i] = self.stimulus_onset_times[i] + on_time
        
        self.total_time = self.stimulus_offset_times[-1]
    
    def _pick_dataset(self) -> str:
        """Randomly pick a dataset based on distribution"""
        return self.config.pick_dataset()
    
    def _pick_image(self, dataset_name: str, target: bool = False) -> Optional[str]:
        """Pick an image from a dataset that hasn't been picked yet"""
        dataset = self.datasets[dataset_name]
        image_bank = dataset.get_target_images() if target else dataset.get_non_target_images()
        
        # Filter out already picked images
        available = [
            filename for filename in image_bank 
            if filename not in self.already_picked_filenames
        ]
        
        if len(available) == 0:
            return None
        
        image_filename = random.choice(available)
        self.already_picked_filenames.add(image_filename)
        return image_filename
    
    def _load_source_framedata(self) -> List[Dict]:
        """Load framedata from an existing JSON file (non-targets only)"""
        print(f"\nLoading source framedata from {self.source_framedata_json}...")
        
        with open(self.source_framedata_json, 'r') as f:
            framedata = json.load(f)

        # Step 0. If the 'target' key is absent, add it to all frames with value False
        for frame in framedata['framedata']:
            if 'target' not in frame:
                frame['target'] = False
            elif type(frame['target']) == str:
                frame['target'] = (frame['target'].lower() == 'true')

        # Filtering step 1. Remove all target frames
        framedata['framedata'] = [frame for frame in framedata['framedata'] if not frame['target']]
        print(f"Filtered {len(framedata['framedata'])} non-target frames")

        # Filtering step 2. Only keep unique frames (which have unique image paths)
        unique_image_paths = set()
        unique_framedata = []
        for frame in framedata['framedata']:
            if frame['image_path'] not in unique_image_paths:
                unique_image_paths.add(frame['image_path'])
                unique_framedata.append(frame)
        framedata['framedata'] = unique_framedata
        print(f"Filtered {len(framedata['framedata'])} unique non-target frames")

        # Clip framedata to N_frames and remove N_target_frames
        n_non_target_needed = min(self.config.N_FRAMES - self.config.N_TARGET_FRAMES, len(framedata['framedata']))
        framedata['framedata'] = random.sample(framedata['framedata'], n_non_target_needed)
        if len(framedata['framedata']) < self.config.N_FRAMES - self.config.N_TARGET_FRAMES:
            print(f"Warning: Not enough non-target frames to meet the required number of frames ({self.config.N_FRAMES - self.config.N_TARGET_FRAMES}). Using {len(framedata['framedata'])} non-target frames instead.")
        print(f"Trimmed framedata to {len(framedata['framedata'])} unique non-target frames (clipped to N_frames and removed N_target_frames; random subset)")
        
        self.N_FRAMES = len(framedata['framedata']) + self.config.N_TARGET_FRAMES
            
        return framedata['framedata']
    
    def _process_source_frame(self, source_frame: Dict, copy_image: bool = True) -> Dict:
        """Process a frame from source JSON - copy images and update paths"""
        # Extract the original image path from source
        source_image_path = source_frame['image_path']
        dataset_name = source_frame['dataset']
        # image_filename = source_frame['image_filename']
        image_filename = source_frame['image_path'].split("/")[-1]

        # XXX: This is a hack-around because the dataset naming is inconsistent between different codebases for the trailer faces HQ dataset. Need to fix and find all points in the codebase where this is a problem.
        source_image_path = source_image_path.replace("TrailerFacesHQ/TrailerFacesHQ/", "tfhqv2/")
        
        # Copy image if requested
        if copy_image:
            save_dir = f"templates/{self.config.TEMPLATE_NAME}/"
            new_image_path = os.path.join(
                save_dir, 
                f"image_datasets/{dataset_name}/{dataset_name}/{image_filename}"
            )
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
            shutil.copy2(source_image_path, new_image_path)
            
            # Update the frame with new path
            processed_frame = source_frame.copy()
            processed_frame['image_path'] = new_image_path
            return processed_frame
        
        return source_frame
    
    def _add_frame(self, target: bool = False, copy_image: bool = True) -> Dict:
        """Add a frame with an image from a randomly selected dataset"""
        image_filename = None
        
        # Try to pick an image until successful
        while image_filename is None:
            dataset_name = self._pick_dataset()
            image_filename = self._pick_image(dataset_name, target=target)
        
        dataset = self.datasets[dataset_name]
        image_path = dataset.get_image_path(image_filename)
        
        # Copy image if requested
        if copy_image:
            save_dir = f"templates/{self.config.TEMPLATE_NAME}/"
            new_image_path = os.path.join(
                save_dir, 
                f"image_datasets/{dataset_name}/{dataset_name}/{image_filename}"
            )
            os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
            # print(f"Copying image from {image_path} to {new_image_path}")
            shutil.copy2(image_path, new_image_path)
            image_path = new_image_path
        
        # Generate random crop
        img = cv2.imread(image_path)
        top, left, size, _ = screen_setup.generate_random_crop(img)
        
        return {
            "dataset": dataset_name,
            "image_filename": image_filename,
            "image_path": image_path,
            "target": target,
            "crop": {
                "top": top,
                "left": left,
                "width": size,
                "height": size,
            }
        }
    
    def generate_frames(self) -> None:
        """Generate all frames for the template"""
        print("\nGenerating the frames...")
        
        # Calculate number of target frames
        n_target_frames = self.config.N_TARGET_FRAMES
        
        # Check if we're using source framedata
        if self.source_framedata_json:
            # Load non-target frames from source JSON
            source_non_target_frames = self._load_source_framedata()

            n_target_frames = self.config.N_FRAMES - len(source_non_target_frames)
            
            # Step 1: Add target frames
            print(f"Generating {n_target_frames} target frames...")
            for i in tqdm(range(n_target_frames), desc="Adding target frames"):
                self.framedata.append(self._add_frame(target=True))
            
            # Step 2: Process and add non-target frames from source
            print(f"Processing {len(source_non_target_frames)} non-target frames from source...")
            for source_frame in tqdm(source_non_target_frames, desc="Processing non-target frames"):
                self.framedata.append(self._process_source_frame(source_frame))
        else:
            # Original logic: generate all frames from scratch
            # Step 1: Add target frames
            for i in tqdm(range(n_target_frames), desc="Adding target frames"):
                self.framedata.append(self._add_frame(target=True))
            
            # Step 2: Add non-target frames
            n_non_target_frames = self.config.N_FRAMES - n_target_frames
            for i in tqdm(range(n_non_target_frames), desc="Adding non-target frames"):
                self.framedata.append(self._add_frame(target=False))
        
        # Step 3: Repeat frames
        self.framedata *= self.config.N_REPEATS
        
        # Step 4: Shuffle frames
        print("Shuffling frames...")
        np.random.shuffle(self.framedata)
        
        # Step 5: Enforce distance constraints between target frames
        # print("Enforcing distance constraints...")
        # self._enforce_target_distance_constraints()
    
    def _enforce_target_distance_constraints(self) -> None:
        """Ensure minimum distance between target frames"""
        pop_image_paths = []
        last_target_frame_i = -100
        
        for i, frame in enumerate(self.framedata):
            if frame['target']:
                if i - last_target_frame_i < self.config.MIN_DISTANCE_BETWEEN_TARGET_FRAMES:
                    pop_image_paths.append(frame['image_path'])
                else:
                    last_target_frame_i = i
        print("Number of target frames to replace using non-target frames: ", len(pop_image_paths))
        
        # Identify frames to replace
        pop_frames_i = [
            i for i, frame in enumerate(self.framedata)
            if frame['image_path'] in pop_image_paths
        ]
        
        # Generate replacement non-target frames
        n_replacements = len(pop_frames_i) // self.config.N_REPEATS
        replacement_frames = [
            self._add_frame(target=False) 
            for _ in range(n_replacements)
        ] * self.config.N_REPEATS
        np.random.shuffle(replacement_frames)
        
        # Replace frames
        for frame_i, replacement in zip(pop_frames_i, replacement_frames):
            self.framedata[frame_i] = replacement
    
    def calculate_dataset_stats(self) -> Dict:
        """Calculate statistics for each dataset"""
        stats = {}
        
        for frame in self.framedata:
            dataset = frame['dataset']
            if dataset not in stats:
                stats[dataset] = {'count': 0, 'target_count': 0}
            stats[dataset]['count'] += 1
            if frame['target']:
                stats[dataset]['target_count'] += 1
        
        # Calculate proportions and normalize by repeats
        for dataset, stat in stats.items():
            stat['proportion'] = round(stat['count'] / len(self.framedata), 2)
            stat['count'] = stat['count'] // self.config.N_REPEATS
            stat['target_count'] = stat['target_count'] // self.config.N_REPEATS
        
        return stats
    
    def build_template(self) -> Dict:
        """Build the complete template dictionary"""
        return {
            "settings": {
                "N_FRAMES": self.config.N_FRAMES,
                "N_REPEATS": self.config.N_REPEATS,
                "TIME_ON_FROM": self.config.TIME_ON_FROM,
                "TIME_ON_TO": self.config.TIME_ON_TO,
                "TIME_OFF_FROM": self.config.TIME_OFF_FROM,
                "TIME_OFF_TO": self.config.TIME_OFF_TO,
                "STIMULUS_ONSET_TIMES_PREDETERMINED": self.stimulus_onset_times.tolist(),
                "STIMULUS_OFFSET_TIMES_PREDETERMINED": self.stimulus_offset_times.tolist(),
                "MIN_DISTANCE_BETWEEN_TARGET_FRAMES": self.config.MIN_DISTANCE_BETWEEN_TARGET_FRAMES,
                "MAX_DISTANCE_BETWEEN_TARGET_FRAMES": self.config.MAX_DISTANCE_BETWEEN_TARGET_FRAMES,
                "TEMPLATE_NAME": self.config.TEMPLATE_NAME,
                "RANDOM_SEED_STRING": self.config.RANDOM_SEED_STRING,
                "RANDOM_SEED": self.config.RANDOM_SEED,
                "TOTAL_TIME": self.total_time,
            },
            "framedata": self.framedata,
            "dataset_stats": self.calculate_dataset_stats(),
        }
    
    def save_template(self, save_dir: str) -> None:
        """Save the template to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Copy this script for reproducibility
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        shutil.copy2(
            __file__, 
            os.path.join(save_dir, f'template_generate_{timestamp}.py')
        )
        
        # Build and save template
        template = self.build_template()
        
        # Print statistics
        n_target = len([f for f in self.framedata if f['target']])
        n_non_target = len([f for f in self.framedata if not f['target']])
        print(f"\nGenerated {len(self.framedata)} frames: "
              f"{n_target} target frames and {n_non_target} non-target frames. "
              f"Saving the template...")
        print(f"Total time of the trial: {self.total_time:.1f} seconds "
              f"({self.total_time/60:.1f} minutes)")
        print("Dataset statistics:", json.dumps(template['dataset_stats'], indent=4))
        
        # Save to file
        template_path = os.path.join(save_dir, "template.json")
        with open(template_path, "w") as f:
            json.dump(template, f, indent=4)
        print("Saved to", template_path)


### MAIN EXECUTION ###

def main(source_framedata_json: Optional[str] = None, template_prefix: Optional[str] = None):
    """Main execution function
    
    Args:
        source_framedata_json: Optional path to existing template JSON file.
                              If provided, non-target frames will be loaded from this file,
                              and only target frames will be generated fresh.
        template_prefix: Required when source_framedata_json is provided. 
                        Custom prefix for the new template name to distinguish it from the source.
    """
    # Validate inputs
    if source_framedata_json and not template_prefix:
        raise ValueError("template_prefix is required when source_framedata_json is provided")
    
    # Create configuration
    config = TemplateConfig(template_prefix=template_prefix)
    
    print("\n", "Generating template:")
    print(f"  Name: {config.TEMPLATE_NAME}")
    print(f"  Frames: {config.N_FRAMES} (x{config.N_REPEATS} repeats)")
    print(f"  Timing: ON={config.TIME_ON_FROM}-{config.TIME_ON_TO}ms, "
          f"OFF={config.TIME_OFF_FROM}-{config.TIME_OFF_TO}ms")
    print(f"  Random seed: {config.RANDOM_SEED_STRING}")
    if source_framedata_json:
        print(f"  Source framedata: {source_framedata_json}")
    print()
    
    # Initialize datasets
    datasets = {}
    
    # ILSVRC2012 (ImageNet)
    ilsvrc_dataset = ILSVRC2012Dataset(
        base_path="image_datasets/ILSVRC2012_img_val/ILSVRC2012_img_val/",
        val_prep_path="image_datasets/ILSVRC2012_img_val/val_prep.sh",
        synset_words_path="image_datasets/ILSVRC2012_img_val/synset_words.txt",
        target_synset='dog.n.01',
        blacklisted_classes=[]
    )
    ilsvrc_dataset.preprocess()
    datasets[ilsvrc_dataset.get_dataset_name()] = ilsvrc_dataset
    
    # OASIS
    oasis_dataset = OASISDataset(
        base_path="image_datasets/OASIS/OASIS",
        csv_path="image_datasets/OASIS/OASIS_bygender_CORRECTED_092617.csv",
        target_prefix="dog",
        blacklist_prefixes=["nude"]
    )
    oasis_dataset.preprocess()
    datasets[oasis_dataset.get_dataset_name()] = oasis_dataset
    
    # FER (Facial expressions)
    fer_dataset = FERDataset(
        base_path="image_datasets/FER/FER"
    )
    fer_dataset.preprocess()
    datasets[fer_dataset.get_dataset_name()] = fer_dataset
    
    # Trailer Faces HQ
    trailer_faces_hq_dataset = TrailerFacesHQDataset(
        base_path="image_datasets/tfhqv2"
    )
    trailer_faces_hq_dataset.preprocess()
    datasets[trailer_faces_hq_dataset.get_dataset_name()] = trailer_faces_hq_dataset

    # Generate template
    generator = TemplateGenerator(config, datasets, source_framedata_json=source_framedata_json)
    generator.generate_frames()
    
    # Save template
    save_dir = f"templates/{config.TEMPLATE_NAME}/"
    generator.save_template(save_dir)


if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    source_json = None
    prefix = None
    
    if len(sys.argv) > 1:
        source_json = sys.argv[1]
        
        # If source JSON is provided, template prefix is required
        if len(sys.argv) < 3:
            print("Error: When providing a source framedata JSON, you must also provide a template prefix.")
            print("\nUsage:")
            print("  Standard mode:  python template_generate.py")
            print("  Reuse mode:     python template_generate.py <source_json_path> <template_prefix>")
            print("\nExample:")
            print("  python template_generate.py templates/first_n480/template.json second")
            sys.exit(1)
        
        prefix = sys.argv[2]
        print(f"Using source framedata from: {source_json}")
        print(f"Template prefix: {prefix}")
    
    main(source_framedata_json=source_json, template_prefix=prefix)
