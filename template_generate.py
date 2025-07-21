import os
import random
import cv2
import numpy as np
import json
import shutil
import time
import screen_setup
import math

### SETTING UP THE GENERAL PARAMETERS ###

N_FRAMES = 1000 # unique frames
N_REPEATS = 2 # number of times to repeat each frame

TIME_ON = 100 # ms
TIME_OFF = 200 # ms

MIN_DISTANCE_BETWEEN_TARGET_FRAMES = math.ceil( 2 * 1000/(TIME_ON+TIME_OFF)) # minimum 2 seconds between target frames
MAX_DISTANCE_BETWEEN_TARGET_FRAMES = math.ceil( 30 * 1000/(TIME_ON+TIME_OFF)) # maximum 20 seconds between target frames. NOTE: this constraint is not enforced strictly, but is roughly guiding the number of target frames

RANDOM_SEED_STRING = "1" # for reproducibility
TEMPLATE_PREFIX = "test"

TEMPLATE_NAME = f"{TEMPLATE_PREFIX}_n{N_FRAMES}_on{TIME_ON}_off{TIME_OFF}_s{RANDOM_SEED_STRING}"

### PICKING THE DATASET AND IMAGE ###

def pick_dataset():
    r = random.random()
    if r < 0.5: # 50% chance to pick ILSVRC2012_img_val
        return "ILSVRC2012_img_val"
    else:
        return "OASIS"

FILENAMES = {} # structure: {dataset: {target: [image_filenames], non_target: [image_filenames]}}
FRAMEDATA = [] # structure: {frame_i: {}} -- history of picked images

ALREADY_PICKED_IMAGE_FILENAMES = set()
def pick_image(dataset: str, target=False):
    assert dataset in FILENAMES, f"Dataset {dataset} not found in FILENAMES"
    assert len(FILENAMES[dataset]["target"]) > 0, f"No target images found for dataset {dataset}"
    assert len(FILENAMES[dataset]["non_target"]) > 0, f"No non-target images found for dataset {dataset}"

    image_bank = FILENAMES[dataset]["target" if target else "non_target"]
    image_bank = [filename for filename in image_bank if filename not in ALREADY_PICKED_IMAGE_FILENAMES]
    if len(image_bank) == 0:
        return None # no more images to pick
    
    image_filename = random.choice(image_bank)
    ALREADY_PICKED_IMAGE_FILENAMES.add(image_filename)
    return image_filename

### SETTING UP THE TEMPLATE AND SAVE DIRECTORY ###

RANDOM_SEED = (int.from_bytes(RANDOM_SEED_STRING.encode(), 'little') * 19241) % (2**32)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TOTAL_TIME = N_FRAMES * (TIME_ON + TIME_OFF) * N_REPEATS / 1000 # seconds

TEMPLATE = {
    "settings": {
        "N_FRAMES": N_FRAMES,
        "N_REPEATS": N_REPEATS,

        "TIME_ON": TIME_ON,
        "TIME_OFF": TIME_OFF,

        "MIN_DISTANCE_BETWEEN_TARGET_FRAMES": MIN_DISTANCE_BETWEEN_TARGET_FRAMES,
        "MAX_DISTANCE_BETWEEN_TARGET_FRAMES": MAX_DISTANCE_BETWEEN_TARGET_FRAMES,

        "TEMPLATE_NAME": TEMPLATE_NAME,

        "RANDOM_SEED_STRING": RANDOM_SEED_STRING,
        "RANDOM_SEED": RANDOM_SEED,

        "TOTAL_TIME": TOTAL_TIME,
    }
}
print("\n", "Generating template:", "\n", json.dumps(TEMPLATE, indent=4), "\n")

SAVE_DIR = f"templates/{TEMPLATE_NAME}/"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_timecode():
    return time.strftime('%Y-%m-%d_%H-%M-%S')
shutil.copy2(__file__, os.path.join(SAVE_DIR, f'template_generate_{get_timecode()}.py')) # Copy this file to the save directory for reproducibility

### IMAGENET VAL CLASS PROCESSING ###

ILSVRC2012_img_val_target_synset = 'dog.n.01'
ILSVRC2012_img_val_BLACKLISTED_CLASSES = []

# Get the class mapping from the ImageNet validation set
def ILSVRC2012_img_val_calculate_class_mapping():
    with open("image_datasets/ILSVRC2012_img_val/val_prep.sh") as f:
        val_prep_lines = [line.strip() for line in f.readlines()]
    val_prep_lines = [line[3:-1] for line in val_prep_lines if line.startswith("mv")]
    filename_synset_mapping = {filename: class_hash for filename, class_hash in [line.split() for line in val_prep_lines]}
    synset_word_mapping = {line.strip()[:9]: line.strip()[10:] for line in open("image_datasets/ILSVRC2012_img_val/synset_words.txt").readlines()}
    filename_word_mapping = {filename: synset_word_mapping[class_hash] for filename, class_hash in filename_synset_mapping.items()}
    return (filename_word_mapping, filename_synset_mapping)
ILSVRC2012_img_val_filename_word_mapping, ILSVRC2012_img_val_filename_synset_mapping = ILSVRC2012_img_val_calculate_class_mapping()

# Get the target synset and its descendants
print("Downloading WordNet for ImageNet labels...")
import nltk
from nltk.data import find
def safe_nltk_download(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource)
safe_nltk_download('wordnet')
from nltk.corpus import wordnet as wn
def get_all_descendants(synset):
    descendants = set()
    frontier = [synset]
    while frontier:
        current = frontier.pop()
        for hyponym in current.hyponyms():
            if hyponym not in descendants:
                descendants.add(hyponym)
                frontier.append(hyponym)
    return descendants

print("")
ILSVRC2012_img_val_target_synset_descendants = get_all_descendants(wn.synset(ILSVRC2012_img_val_target_synset))
ILSVRC2012_img_val_target_synset_descendants_ids = [f"n{syn.offset():08d}" for syn in ILSVRC2012_img_val_target_synset_descendants]
ILSVRC2012_img_val_TARGET_CLASSES = ILSVRC2012_img_val_target_synset_descendants_ids

ILSVRC2012_img_val_TARGET_IMAGE_FILENAMES = [
    filename for filename in ILSVRC2012_img_val_filename_synset_mapping.keys()
    if ILSVRC2012_img_val_filename_synset_mapping[filename] in ILSVRC2012_img_val_TARGET_CLASSES
]
ILSVRC2012_img_val_NONTARGET_IMAGE_FILENAMES = [
    filename for filename in ILSVRC2012_img_val_filename_synset_mapping.keys()
    if ILSVRC2012_img_val_filename_synset_mapping[filename] not in ILSVRC2012_img_val_TARGET_CLASSES and
    ILSVRC2012_img_val_filename_synset_mapping[filename] not in ILSVRC2012_img_val_BLACKLISTED_CLASSES
]

FILENAMES["ILSVRC2012_img_val"] = {
    "target": ILSVRC2012_img_val_TARGET_IMAGE_FILENAMES,
    "non_target": ILSVRC2012_img_val_NONTARGET_IMAGE_FILENAMES
}

print(f"ILSVRC2012_img_val: Found {len(ILSVRC2012_img_val_TARGET_IMAGE_FILENAMES)} target images and {len(ILSVRC2012_img_val_NONTARGET_IMAGE_FILENAMES)} non-target images")

### OASIS CLASS PROCESSING ###

import pandas as pd
def load_oasis_csv_as_dataframe(file_path):
    return pd.read_csv(file_path)
oasis_df = load_oasis_csv_as_dataframe('image_datasets/OASIS/OASIS_bygender_CORRECTED_092617.csv')

OASIS_ALL_IMAGE_FILENAMES = [filename.strip() + ".jpg" for filename in oasis_df['Theme'].tolist()]

OASIS_BLACKLISTED_IMAGE_FILENAMES = [
    filename for filename in OASIS_ALL_IMAGE_FILENAMES
    if filename.lower().startswith("nude")
]

OASIS_TARGET_IMAGE_FILENAMES = [
    filename for filename in OASIS_ALL_IMAGE_FILENAMES
    if filename.lower().startswith("dog")
]

OASIS_NONTARGET_IMAGE_FILENAMES = [
    filename for filename in OASIS_ALL_IMAGE_FILENAMES
    if filename not in OASIS_TARGET_IMAGE_FILENAMES and
    filename not in OASIS_BLACKLISTED_IMAGE_FILENAMES
]

FILENAMES["OASIS"] = {
    "target": OASIS_TARGET_IMAGE_FILENAMES,
    "non_target": OASIS_NONTARGET_IMAGE_FILENAMES
}

print(f"OASIS: Found {len(OASIS_TARGET_IMAGE_FILENAMES)} target images and {len(OASIS_NONTARGET_IMAGE_FILENAMES)} non-target images")

### PICKING THE IMAGE ###

print("\nGenerating the frames...") 
min_n_targets_frames = N_FRAMES // MAX_DISTANCE_BETWEEN_TARGET_FRAMES # minimum number of target frames
max_n_targets_frames = N_FRAMES // MIN_DISTANCE_BETWEEN_TARGET_FRAMES # maximum number of target frames
n_target_frames = random.randint(min_n_targets_frames, max_n_targets_frames)

def add_frame(target=False):
    image_filename = None
    while image_filename is None: # try to pick an image until it succeeds (can only fail if all images from the dataset have been picked)
        dataset = pick_dataset()
        image_filename = pick_image(dataset, target=target)
    image_path = f"image_datasets/{dataset}/{dataset}/{image_filename}"

    img = cv2.imread(image_path)
    top, left, size, size = screen_setup.generate_random_crop(img)
    return {
        "dataset": dataset,
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

# Step 1. Add target and non-target frames (order is not important), for the correct number of repeats
for frame_i in range(n_target_frames):
    if frame_i % 100 == 0 or frame_i == n_target_frames - 1:
        print(f"Adding target frames: {frame_i+1}/{n_target_frames}")
    FRAMEDATA.append(add_frame(target=True))
for frame_i in range(N_FRAMES - n_target_frames):
    if frame_i % 100 == 0 or frame_i == N_FRAMES - n_target_frames - 1:
        print(f"Adding non-target frames: {frame_i+1}/{N_FRAMES - n_target_frames}")
    FRAMEDATA.append(add_frame(target=False))
FRAMEDATA *= N_REPEATS

# Step 2. Shuffle the frames
np.random.shuffle(FRAMEDATA)

# Step 3. Go through the frames to ensure that the distance between target frames is within the constraints.
# replace all the target frames that violate the constraints with non-target frames
pop_image_paths = []
last_target_frame_i = 0
for frame_i in range(len(FRAMEDATA)):
    if FRAMEDATA[frame_i]['target']:
        last_target_frame_i = frame_i
    else:
        if frame_i - last_target_frame_i < MIN_DISTANCE_BETWEEN_TARGET_FRAMES:
            pop_image_paths.append(FRAMEDATA[frame_i]['image_path'])
pop_frames_i = []
for frame_i in range(len(FRAMEDATA)):
    if FRAMEDATA[frame_i]['image_path'] in pop_image_paths:
        pop_frames_i.append(frame_i)
nontarget_replacement_frames = [add_frame(target=False) for _ in range(len(pop_frames_i)//N_REPEATS)] * N_REPEATS
np.random.shuffle(nontarget_replacement_frames)
for frame_i, replacement_frame in zip(pop_frames_i, nontarget_replacement_frames):
    FRAMEDATA[frame_i] = replacement_frame

### SAVE THE TEMPLATE ###

# Calculate statistics for each dataset
dataset_stats = {}
for frame in FRAMEDATA:
    dataset = frame['dataset']
    if dataset not in dataset_stats:
        dataset_stats[dataset] = {'count': 0, 'target_count': 0}
    dataset_stats[dataset]['count'] += 1
    if frame['target']:
        dataset_stats[dataset]['target_count'] += 1
# Calculate percentages and add to TEMPLATE
for dataset, stats in dataset_stats.items():
    stats['proportion'] = round((stats['count'] / len(FRAMEDATA)), 2)
    stats['count'] = stats['count'] // N_REPEATS
    stats['target_count'] = stats['target_count'] // N_REPEATS

TEMPLATE["framedata"] = FRAMEDATA
TEMPLATE["dataset_stats"] = dataset_stats

# Print statistics
print(f"\nGenerated {len(FRAMEDATA)} frames: {len([frame for frame in FRAMEDATA if frame['target']])} target frames and {len([frame for frame in FRAMEDATA if not frame['target']])} non-target frames. Saving the template...")
print(f"Total time: of the trial: {TOTAL_TIME:.1f} seconds ({TOTAL_TIME/60:.1f} minutes)")
print("Dataset statistics:", json.dumps(dataset_stats, indent=4))

# Save TEMPLATE to file
with open(os.path.join(SAVE_DIR, "template.json"), "w") as f:
    json.dump(TEMPLATE, f, indent=4)
print("Saved to", os.path.join(SAVE_DIR, "template.json"))


### OPTIONAL: PREPROCESS THE FRAMES ###

# image_frames = []
# for frame_i, framedata in enumerate(FRAMEDATA):
#     if frame_i % 100 == 0 or frame_i == len(FRAMEDATA) - 1:
#         print(f"Preparing frame {frame_i+1} of {len(FRAMEDATA)}")
#     image_frames.append(screen_setup.prepare_frame(framedata, cross=True, square=True))
# image_frames = np.array(image_frames)
# np.save(os.path.join(SAVE_DIR, "image_frames.npy"), image_frames)
# print("Saved to", os.path.join(SAVE_DIR, "image_frames.npy"))