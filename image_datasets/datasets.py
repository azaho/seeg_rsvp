"""
Dataset Classes for SEEG RSVP Experiments

This module contains dataset interface and implementations for:
- ILSVRC2012 (ImageNet validation set)
- OASIS
- FER (Facial Expression Recognition)
"""

import os
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import nltk
from nltk.data import find
from nltk.corpus import wordnet as wn


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


class TrailerFacesHQDataset(Dataset):
    """Trailer Faces HQ dataset"""
    
    def __init__(self, base_path: str):
        super().__init__(base_path)
        
    def get_dataset_name(self) -> str:
        return "TrailerFacesHQ"
    
    def preprocess(self) -> None:
        """Preprocess Trailer Faces HQ dataset"""
        # Get all image files in the base path
        self.image_files = [f for f in os.listdir(self.base_path) if f.endswith(('.jpg',))]
        print(f"TrailerFacesHQ: Found {len(self.image_files)} image files")

        self.non_target_images = self.image_files
        self.target_images = []


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

        missed_target_images = ['ILSVRC2012_val_00035676.JPEG'] # manually added
        
        # Separate target and non-target images
        self.target_images = [
            filename for filename in filename_synset_mapping.keys()
            if filename_synset_mapping[filename] in target_classes or filename in missed_target_images
        ]
        
        self.non_target_images = [
            filename for filename in filename_synset_mapping.keys()
            if (filename_synset_mapping[filename] not in target_classes and
                filename_synset_mapping[filename] not in self.blacklisted_classes and
                filename not in missed_target_images)
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




