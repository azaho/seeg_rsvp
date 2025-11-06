
To install the necessary requirements and packages, please run the following commands. First, optionally, install a virtual environment with:
```python
python -m venv .venv
source .venv/bin/activate # On Windows: .venv/Scripts/activate
pip install --upgrade pip
```
Then, use `pip` to install the necessary packages to run the code in this repository:
```python
pip install -r requirements.txt
```

# Template Generation Usage Guide

## Overview

The `template_generate.py` script now supports two modes of operation:

1. **Standard Mode**: Generate all frames (targets and non-targets) from scratch
2. **Reuse Mode**: Reuse non-target frames from an existing template, generate fresh target frames

## Standard Mode (Original Functionality)

Generate a complete template from scratch:

```bash
python template_generate.py
```

This will:
- Generate both target and non-target frames from the configured datasets
- Apply timing constraints and distance constraints
- Shuffle and save the template

## Reuse Mode (New Functionality)

Use existing non-target frames and generate fresh target frames.

**Important:** When providing a source JSON, you MUST also provide a template prefix to distinguish the new template.

```bash
python template_generate.py templates/first_n480_on100-100_off125-175_s4/template.json second
```

This will:
- Load non-target frames from the specified template JSON
- Generate fresh target frames based on the current config settings
- Combine targets and non-targets
- Shuffle all frames with the new random seed
- Copy images to the new template directory
- Apply distance constraints between target frames
- Save the new template with updated timing

## Configuration

Edit the `TemplateConfig` class in `template_generate.py` to adjust:

- `N_FRAMES`: Number of unique frames
- `N_REPEATS`: Number of repetitions
- `TIME_ON_FROM/TO`: Stimulus on-time range (ms)
- `TIME_OFF_FROM/TO`: Stimulus off-time range (ms)
- `RANDOM_SEED_STRING`: Seed for reproducibility
- `MIN/MAX_DISTANCE_BETWEEN_TARGET_FRAMES`: Constraint on target spacing

## Notes

- **Template prefix is REQUIRED** when using reuse mode (providing source JSON)
- The reuse mode preserves crop information from source frames
- Image paths are updated to the new template directory
- All images are copied to maintain template independence
- Target frame count is calculated based on config constraints
- The new template will be saved with the custom prefix in its name
- Example: prefix "second" → `second_n480_on100-100_off125-175_s4`

# Game/Experiment Usage Guide

## Overview

The `game.py` script runs the RSVP (Rapid Serial Visual Presentation) experiment. It displays a stream of images where participants must click when they see a target (dog).

## Running the Experiment

To run an experiment with a specific template:

```bash
python game.py <template_name>
```

For example:

```bash
python game.py first_n10_on100-100_off125-175_s4
```

This will:
- Load the template from `templates/first_n10_on100-100_off125-175_s4/template.json`
- Create a trial data directory with timestamp in `trial_data/`
- Copy the config and template files to the trial directory
- Start the experiment

## Experiment Flow

1. **Instruction Screen**: Displays instructions and waits for mouse click or key press to begin
2. **RSVP Stream**: Rapidly presents images with targets (dogs) intermixed
3. **Results Screen**: Shows performance metrics (targets caught, accuracy, medal)

## Controls

### During Experiment
- **Mouse Click** (Left/Right): Register a target detection
- **SPACE**: Alternative to mouse click
- **S**: Start the stream (from initial gray screen)
- **P**: Pause/Resume the stream
- **T**: Send trigger event (for synchronization with external systems)
- **Q**: Quit the experiment

### Results Screen
- **Q**: Quit and close the application

## Data Output

The experiment saves data to `trial_data/<timestamp>/`:

1. **events/**: Directory containing event logs
   - `events_<timestamp>.json`: All events (clicks, stimulus onsets, etc.)

2. **log_arrays.npz**: NumPy archive containing:
   - `CLICK_TIMES`: Timestamps of all clicks
   - `SUCCESS_CLICKS`: Timestamps of successful target detections
   - `FAILURE_CLICKS`: Timestamps of false alarms
   - `STIMULUS_ONSET_TIMES_PREDETERMINED`: Planned stimulus onset times
   - `STIMULUS_OFFSET_TIMES_PREDETERMINED`: Planned stimulus offset times
   - `STIMULUS_ONSET_TIMES_ACTUAL`: Actual stimulus onset times
   - `STIMULUS_OFFSET_TIMES_ACTUAL`: Actual stimulus offset times
   - `TARGET_STIMULUS_IDX`: Indices of target stimuli

3. **config_<timestamp>.py**: Copy of the configuration used
4. **template.json**: Copy of the template used

## Configuration

Edit `config.py` to adjust:
- `SCREEN_ID`: Which monitor to use (-1 for last monitor)
- `SCREEN_WIDTH/HEIGHT`: Display resolution
- `VISUAL_STIMULUS_WIDTH/HEIGHT`: Size of the stimulus area
- `VISUAL_SQUARE_SIZE`: Size of the corner square indicator
- `BACKUP_CONFIG_INTERVAL`: How often to save events during the experiment

## Performance Feedback

The results screen shows:
- **Targets caught**: Number and total (e.g., "15 / 20")
- **Medal**: Gold (≥90%), Silver (≥80%), Bronze (≥50%)

Configurable display options in `show_results_screen()`:
- `show_targets_caught`: Display targets caught count
- `show_targets_missed`: Display missed targets
- `show_wrong_clicks`: Display false alarm count
- `show_percentage`: Display accuracy percentage
- `show_medal`: Display medal earned

## Requirements

Before running, ensure:
1. A valid template exists in `templates/<template_name>/`
2. Sound files exist: `sound_success.mp3` and `sound_failure.mp3`
3. The `config.py` file is properly configured for your display setup
4. Required packages are installed (pygame, opencv-python, numpy)

## Notes

- The experiment runs in fullscreen mode on the specified display
- Screensaver is disabled during the experiment
- Mouse cursor is hidden during the experiment
- Target detection window is 2 seconds after stimulus onset
- Events are periodically backed up during the experiment
