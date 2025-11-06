import os
import random
import cv2
import numpy as np
import json
import shutil
import time
import screen_setup
import math
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import nltk
from nltk.data import find
from nltk.corpus import wordnet as wn


### CONFIGURATION ###

class TemplateConfig:
    """Configuration for template generation"""
    def __init__(self):
        # Frame settings
        self.N_FRAMES = 10
        self.N_REPEATS = 4
        
        # Timing settings (in ms)
        self.TIME_ON_FROM = 100
        self.TIME_ON_TO = 100
        self.TIME_OFF_FROM = 125
        self.TIME_OFF_TO = 175
        
        # Derived timing
        self.TIME_ON_MEAN = (self.TIME_ON_FROM + self.TIME_ON_TO) / 2
        self.TIME_OFF_MEAN = (self.TIME_OFF_FROM + self.TIME_OFF_TO) / 2
        
        # Distance constraints
        self.MIN_DISTANCE_BETWEEN_TARGET_FRAMES = math.ceil(
            2 * 1000 / (self.TIME_ON_MEAN + self.TIME_OFF_MEAN)
        )
        self.MAX_DISTANCE_BETWEEN_TARGET_FRAMES = math.ceil(
            30 * 1000 / (self.TIME_ON_MEAN + self.TIME_OFF_MEAN)
        )
        
        # Random seed and naming
        self.RANDOM_SEED_STRING = "4"
        self.TEMPLATE_PREFIX = "first"
        
        # Derived properties
        self.TEMPLATE_NAME = (
            f"{self.TEMPLATE_PREFIX}_n{self.N_FRAMES}_"
            f"on{self.TIME_ON_FROM}-{self.TIME_ON_TO}_"
            f"off{self.TIME_OFF_FROM}-{self.TIME_OFF_TO}_"
            f"s{self.RANDOM_SEED_STRING}"
        )
        self.RANDOM_SEED = (
            int.from_bytes(self.RANDOM_SEED_STRING.encode(), 'little') * 19241
        ) % (2**32)


### DATASET INTERFACE ###

class Dataset(ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.target_images: List[str] = []
        self.non_target_images: List[str] = []
        
    @abstractmethod
    def preprocess(self) -> None:
        """Load and preprocess the dataset"""
        pass
    
    def get_target_images(self) -> List[str]:
        """Return list of target image filenames"""
        return self.target_images
    
    def get_non_target_images(self) -> List[str]:
        """Return list of non-target image filenames"""
        return self.non_target_images
    
    def get_image_path(self, filename: str) -> str:
        """Get full path to an image file"""
        return os.path.join(self.base_path, filename)
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the name of this dataset"""
        pass


class ILSVRC2012Dataset(Dataset):
    """ImageNet ILSVRC2012 validation set dataset"""
    
    def __init__(self, base_path: str, val_prep_path: str, synset_words_path: str, target_synset: str = 'dog.n.01', 
                 blacklisted_classes: Optional[List[str]] = None):
        super().__init__(base_path)
        self.target_synset = target_synset
        self.blacklisted_classes = blacklisted_classes or []
        self.val_prep_path = val_prep_path
        self.synset_words_path = synset_words_path
        
    def get_dataset_name(self) -> str:
        return "ILSVRC2012_img_val"
    
    def _calculate_class_mapping(self) -> Tuple[Dict, Dict]:
        """Calculate filename to class mapping from val_prep.sh"""
        with open(self.val_prep_path) as f:
            val_prep_lines = [line.strip() for line in f.readlines()]
        val_prep_lines = [line[3:-1] for line in val_prep_lines if line.startswith("mv")]
        filename_synset_mapping = {
            filename: class_hash 
            for filename, class_hash in [line.split() for line in val_prep_lines]
        }
        
        synset_file = self.synset_words_path
        synset_word_mapping = {
            line.strip()[:9]: line.strip()[10:] 
            for line in open(synset_file).readlines()
        }
        
        filename_word_mapping = {
            filename: synset_word_mapping[class_hash] 
            for filename, class_hash in filename_synset_mapping.items()
        }
        
        return filename_word_mapping, filename_synset_mapping
    
    def _get_synset_descendants(self, synset_name: str) -> List[str]:
        """Get all descendant synsets for a given synset"""
        synset = wn.synset(synset_name)
        descendants = set()
        frontier = [synset]
        
        while frontier:
            current = frontier.pop()
            for hyponym in current.hyponyms():
                if hyponym not in descendants:
                    descendants.add(hyponym)
                    frontier.append(hyponym)
        
        return [f"n{syn.offset():08d}" for syn in descendants]
    
    def preprocess(self) -> None:
        """Preprocess ImageNet dataset"""
        # Ensure WordNet is available
        try:
            find('corpora/wordnet')
        except LookupError:
            print("Downloading WordNet for ImageNet labels...")
            nltk.download('wordnet')
        
        # Get class mappings
        _, filename_synset_mapping = self._calculate_class_mapping()
        
        # Get target classes (synset descendants)
        target_classes = self._get_synset_descendants(self.target_synset)
        
        # Separate target and non-target images
        self.target_images = [
            filename for filename in filename_synset_mapping.keys()
            if filename_synset_mapping[filename] in target_classes
        ]
        
        self.non_target_images = [
            filename for filename in filename_synset_mapping.keys()
            if (filename_synset_mapping[filename] not in target_classes and
                filename_synset_mapping[filename] not in self.blacklisted_classes)
        ]
        
        print(f"ILSVRC2012_img_val: Found {len(self.target_images)} target images "
              f"and {len(self.non_target_images)} non-target images")


class OASISDataset(Dataset):
    """OASIS dataset"""
    
    def __init__(self, base_path: str, csv_path: str, 
                 target_prefix: str = "dog", blacklist_prefixes: Optional[List[str]] = None):
        super().__init__(base_path)
        self.csv_path = csv_path
        self.target_prefix = target_prefix
        self.blacklist_prefixes = blacklist_prefixes or ["nude"]
        
    def get_dataset_name(self) -> str:
        return "OASIS"
    
    def preprocess(self) -> None:
        """Preprocess OASIS dataset"""
        df = pd.read_csv(self.csv_path)
        all_filenames = [filename.strip() + ".jpg" for filename in df['Theme'].tolist()]
        
        # Blacklisted images
        blacklisted = [
            filename for filename in all_filenames
            if any(filename.lower().startswith(prefix.lower()) 
                   for prefix in self.blacklist_prefixes)
        ]
        
        # Target images
        self.target_images = [
            filename for filename in all_filenames
            if filename.lower().startswith(self.target_prefix.lower())
        ]
        
        # Non-target images
        self.non_target_images = [
            filename for filename in all_filenames
            if filename not in self.target_images and filename not in blacklisted
        ]
        
        print(f"OASIS: Found {len(self.target_images)} target images "
              f"and {len(self.non_target_images)} non-target images")


class FERDataset(Dataset):
    """Facial Expression Recognition (FER) dataset"""
    
    def __init__(self, base_path: str, emotions: Optional[List[str]] = None):
        super().__init__(base_path)
        self.emotions = emotions or ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
    def get_dataset_name(self) -> str:
        return "FER"
    
    def preprocess(self) -> None:
        """Preprocess FER dataset"""
        emotion_filenames = {emotion: [] for emotion in self.emotions}
        
        for emotion in self.emotions:
            emotion_dir = os.path.join(self.base_path, "validation", emotion)
            for filename in os.listdir(emotion_dir):
                full_path = os.path.join("validation", emotion, filename)
                emotion_filenames[emotion].append(full_path)
        
        # FER has no target images by default (only non-target faces)
        self.target_images = []
        self.non_target_images = [
            filename 
            for filenames in emotion_filenames.values() 
            for filename in filenames
        ]
        
        emotion_counts = ", ".join([
            f"{len(emotion_filenames[emotion])} {emotion}" 
            for emotion in self.emotions
        ])
        print(f"FER: Found {emotion_counts}")
        print(f"\t Total target images: {len(self.target_images)}; "
              f"Total non-target images: {len(self.non_target_images)}")


### TEMPLATE GENERATOR ###

class TemplateGenerator:
    """Main class for generating experiment templates"""
    
    def __init__(self, config: TemplateConfig, datasets: Dict[str, Dataset]):
        self.config = config
        self.datasets = datasets
        self.already_picked_filenames: set = set()
        self.framedata: List[Dict] = []
        
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
        self.stimulus_offset_times[0] = (time_on_mean + time_off_mean) / 1000
        
        for i in range(1, n_stimuli):
            on_time = random.uniform(self.config.TIME_ON_FROM, self.config.TIME_ON_TO) / 1000
            off_time = random.uniform(self.config.TIME_OFF_FROM, self.config.TIME_OFF_TO) / 1000
            
            self.stimulus_onset_times[i] = self.stimulus_offset_times[i-1] + on_time
            self.stimulus_offset_times[i] = self.stimulus_onset_times[i] + off_time
        
        self.total_time = self.stimulus_offset_times[-1]
    
    def _pick_dataset(self) -> str:
        """Randomly pick a dataset based on distribution"""
        r = random.random()
        if r < 0.4:
            return "ILSVRC2012_img_val"
        elif r < 0.8:
            return "OASIS"
        else:
            return "FER"
    
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
        min_targets = self.config.N_FRAMES // self.config.MAX_DISTANCE_BETWEEN_TARGET_FRAMES
        max_targets = self.config.N_FRAMES // self.config.MIN_DISTANCE_BETWEEN_TARGET_FRAMES
        n_target_frames = random.randint(min_targets, max_targets)
        
        # Step 1: Add target frames
        for i in range(n_target_frames):
            if i % 100 == 0 or i == n_target_frames - 1:
                print(f"Adding target frames: {i+1}/{n_target_frames}")
            self.framedata.append(self._add_frame(target=True))
        
        # Step 2: Add non-target frames
        n_non_target_frames = self.config.N_FRAMES - n_target_frames
        for i in range(n_non_target_frames):
            if i % 100 == 0 or i == n_non_target_frames - 1:
                print(f"Adding non-target frames: {i+1}/{n_non_target_frames}")
            self.framedata.append(self._add_frame(target=False))
        
        # Step 3: Repeat frames
        self.framedata *= self.config.N_REPEATS
        
        # Step 4: Shuffle frames
        np.random.shuffle(self.framedata)
        
        # Step 5: Enforce distance constraints between target frames
        self._enforce_target_distance_constraints()
    
    def _enforce_target_distance_constraints(self) -> None:
        """Ensure minimum distance between target frames"""
        pop_image_paths = []
        last_target_frame_i = 0
        
        for i, frame in enumerate(self.framedata):
            if frame['target']:
                last_target_frame_i = i
            else:
                if i - last_target_frame_i < self.config.MIN_DISTANCE_BETWEEN_TARGET_FRAMES:
                    pop_image_paths.append(frame['image_path'])
        
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

def main():
    """Main execution function"""
    # Create configuration
    config = TemplateConfig()
    
    print("\n", "Generating template:")
    print(f"  Name: {config.TEMPLATE_NAME}")
    print(f"  Frames: {config.N_FRAMES} (x{config.N_REPEATS} repeats)")
    print(f"  Timing: ON={config.TIME_ON_FROM}-{config.TIME_ON_TO}ms, "
          f"OFF={config.TIME_OFF_FROM}-{config.TIME_OFF_TO}ms")
    print(f"  Random seed: {config.RANDOM_SEED_STRING}\n")
    
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
    
    # Generate template
    generator = TemplateGenerator(config, datasets)
    generator.generate_frames()
    
    # Save template
    save_dir = f"templates/{config.TEMPLATE_NAME}/"
    generator.save_template(save_dir)


if __name__ == "__main__":
    main()
