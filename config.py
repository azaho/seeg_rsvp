import numpy as np

### SCREEN SETTINGS

SCREEN_ID = 0
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

### VISUAL STIMULUS SETTINGS

VISUAL_STIMULUS_SIZE = 0.75 # proportion of the screen height

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