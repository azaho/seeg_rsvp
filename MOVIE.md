# Movie Player with Visual Markers

This program plays an MP4 video fullscreen with a white square appearing every N frames in the bottom-right corner.

## Usage

```bash
python game.py <path_to_mp4_file> [square_interval]
```

### Arguments

- `path_to_mp4_file` (required): Path to the MP4 video file you want to play
- `square_interval` (optional): Show white square every N frames (default: 30)

### Examples

```bash
# Play video.mp4 with square every 30 frames (default)
python game.py video.mp4

# Play video.mp4 with square every 60 frames
python game.py video.mp4 60

# Play with full path
python game.py C:\Videos\myvideo.mp4 45
```

## Controls

- **SPACE** or **P**: Toggle play/pause
- **S**: Start playback
- **T**: Create a trigger event (logged with current frame number)
- **Mouse Click** (left or right): Log click event with frame number and position
- **Q** or **ESC**: Quit the application

## Features

- Fullscreen playback
- Automatic video scaling to fit screen while maintaining aspect ratio
- White square marker in bottom-right corner every N frames
- Event logging (saved to `trial_data/` directory)
- Playback progress display

## Configuration

You can modify settings in `config.py`:
- `SCREEN_ID`: Which monitor to use (-1 for last/connected monitor)
- `VISUAL_SQUARE_SIZE`: Size of the white square (proportion of screen height)
- `VISUAL_SQUARE_COLOR`: Color of the square (default: white)

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python
- pygame
- numpy

