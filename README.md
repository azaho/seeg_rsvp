
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

