import numpy as np
import math

### SCREEN SETTINGS

SCREEN_ID = -1 # of <0, use the last monitor (which will be the connected monitor if one is connected)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCREEN_INCHES = 21.5 # inches

### VISUAL STIMULUS SETTINGS

DISTANCE_MONITOR_TO_PATIENT = 0.75 # meters
STIMULUS_ANGULAR_SIZE = 8 # degrees

_monitor_height = 1/100 * 2.54 * SCREEN_INCHES * SCREEN_HEIGHT / (SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)**0.5 # in meters
VISUAL_STIMULUS_SIZE = STIMULUS_ANGULAR_SIZE / 180 * math.pi * DISTANCE_MONITOR_TO_PATIENT / _monitor_height # Automatically calculated based on desired angular size and distance to patient

# Uncomment for a fixed stimulus size
# VISUAL_STIMULUS_SIZE = 0.75 # proportion of the screen height

VISUAL_CROSS_SIZE = 1/64  # proportion of the stimulus size
VISUAL_CROSS_THICKNESS = 1/720
VISUAL_CROSS_COLOR = (255, 255, 255)
VISUAL_CROSS_BACKGROUND = True
VISUAL_CROSS_BACKGROUND_COLOR = (0, 0, 0)

VISUAL_SQUARE_SIZE = 120/1080  # proportion of the stimulus size
VISUAL_SQUARE_COLOR = (255, 255, 255)

VISUAL_GRAY_COLOR = np.array([[[128, 128, 128]]]).astype(np.uint8)

VISUAL_STIMULUS_WIDTH = int(min(SCREEN_WIDTH, SCREEN_HEIGHT) * VISUAL_STIMULUS_SIZE)
VISUAL_STIMULUS_HEIGHT = VISUAL_STIMULUS_WIDTH

### OTHER SETTINGS

BACKUP_CONFIG_INTERVAL = 10 # seconds

GAME_FPS = 10


### UTILITY FUNCTIONS

if __name__ == "__main__":
    import pygame
    # Get the list of connected displays
    pygame.init()
    display_info = pygame.display.get_desktop_sizes()

    for i, (width, height) in enumerate(display_info):
        print(f"Display {i}: {width}x{height}")