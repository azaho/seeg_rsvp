import json
import pygame
import os
import screen_setup
from config import *
import time
import shutil
import sys
import cv2

template_name = sys.argv[1] if len(sys.argv) > 1 else "test"
template_path = f"templates/{template_name}/template.json"
assert os.path.exists(template_path), f"Template {template_name} not found."
with open(template_path, 'r') as f:
    TEMPLATE = json.load(f)

EVENTS = []
def create_event(event_type, timestamp=None, **kwargs):
    if timestamp is None:
        timestamp = time.time()
    event = {
        "type": event_type,
        "time": timestamp,
        "meta": kwargs,
    }
    EVENTS.append(event)
    return event
create_event("application_start", template_name=template_name)

### SETTING UP THE SAVE DIRECTORY ###

def get_timecode():
    return time.strftime('%Y-%m-%d_%H-%M-%S')

SAVE_DIR = f"trial_data/{get_timecode()}/"
os.makedirs(SAVE_DIR, exist_ok=True)

shutil.copy2('config.py', os.path.join(SAVE_DIR, f'config_{get_timecode()}.py'))
shutil.copy2(template_path, os.path.join(SAVE_DIR, f'template.json'))

EVENTS_SAVE_DIR = os.path.join(SAVE_DIR, 'events')
os.makedirs(EVENTS_SAVE_DIR, exist_ok=True)
def save_events():
    path = os.path.join(EVENTS_SAVE_DIR, f'events_{get_timecode()}.json')
    with open(path, 'w') as f:
        json.dump(EVENTS, f)
    print(f"Saved events to {path}")
save_events()

### SETTING UP PYGAME ###

pygame.init()

desktop_sizes = pygame.display.get_desktop_sizes()
if SCREEN_ID < 0: SCREEN_ID = len(desktop_sizes) - 1

monitor_width, monitor_height = desktop_sizes[SCREEN_ID]
assert monitor_width == SCREEN_WIDTH and monitor_height == SCREEN_HEIGHT, \
    f"Monitor size does not match screen size in config.py. Please change the SCREEN_WIDTH and SCREEN_HEIGHT" + \
    f" in config.py to match the monitor size, or connect to the correct monitor. Monitor size: {monitor_width}x{monitor_height}, " + \
    f"Screen size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}"

screen = pygame.display.set_mode((monitor_width, monitor_height), pygame.NOFRAME | pygame.FULLSCREEN, display=SCREEN_ID)
pygame.display.set_caption("Frame Viewer")
pygame.display.set_allow_screensaver(False)  # Prevent screensaver from activating
# os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'  # Position window at top-left
pygame.event.set_grab(True)  # Keep input focus
pygame.mouse.set_visible(False)  # Hide the cursor

### SETTING UP THE FRAMES ###

def to_pygame_frame(frame):
    return pygame.image.frombuffer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes(), (frame.shape[1], frame.shape[0]), 'RGB')

gray_frame = to_pygame_frame(screen_setup.get_gray_frame(cross=True, square=False))
gray_frame_square = to_pygame_frame(screen_setup.get_gray_frame(cross=True, square=True))

image_frames = []
for frame_i, framedata in enumerate(TEMPLATE["framedata"]):
    if frame_i % 100 == 0 or frame_i == len(TEMPLATE["framedata"]) - 1:
        print(f"Preparing frame {frame_i+1} of {len(TEMPLATE['framedata'])}")
    image_frames.append(to_pygame_frame(screen_setup.preprocess_image(framedata, cross=True)))

screen.blit(gray_frame, (0, 0))
pygame.display.flip()

### GAME LOOP ###

def blit_image(image_i, square=True):
    x_offset = (SCREEN_WIDTH - VISUAL_STIMULUS_WIDTH) // 2
    y_offset = (SCREEN_HEIGHT - VISUAL_STIMULUS_HEIGHT) // 2
    screen.blit(image_frames[image_i], (x_offset, y_offset))

    if square:
        h, w = SCREEN_HEIGHT, SCREEN_WIDTH
        size = int(VISUAL_SQUARE_SIZE * SCREEN_HEIGHT)
        top = h - size
        right = w - size
        pygame.draw.rect(screen, VISUAL_SQUARE_COLOR, (right, top, size, size))
    pygame.display.flip()

def blit_gray_frame(square=False):
    screen.blit(gray_frame if not square else gray_frame_square, (0, 0))
    pygame.display.flip()

clock = pygame.time.Clock()
paused = True
started = False
current_image_i = 0

click_times = []
stimulus_times = np.zeros(len(TEMPLATE["framedata"]))
target_stimulus_idx = [i for i, framedata in enumerate(TEMPLATE["framedata"]) if framedata["target"]]

time_start_showing_gray_frame = None
def show_stimulus(image_i):
    global time_start_showing_gray_frame

    time_start_showing_stimulus = time.time()
    blit_image(image_i)
    create_event("show_stimulus", image_i=image_i)
    stimulus_times[image_i] = time.time()

    pygame.time.delay(TEMPLATE["settings"]["TIME_ON"] - int((time.time() - time_start_showing_stimulus) * 1000))

    time_start_showing_gray_frame = time.time()
    blit_gray_frame(square=False)

def get_last_target_stimulus_idx(current_image_i):
    # Find the last target stimulus index that is <= current_image_i
    for idx in reversed(target_stimulus_idx):
        if idx <= current_image_i:
            return idx
    return None

running = True
last_backup = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                started = True
                paused = False
            elif event.key == pygame.K_p:
                paused = not paused
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print(f"Mouse clicked at position: {pygame.mouse.get_pos()}")
            create_event("mouse_click")

            success_sound = False
            current_time = time.time()
            last_target_stimulus_idx = get_last_target_stimulus_idx(current_image_i)
            if (last_target_stimulus_idx is not None) and \
                (current_time - stimulus_times[last_target_stimulus_idx] < 2) and \
                (len(click_times) == 0 or (stimulus_times[last_target_stimulus_idx] > click_times[-1])): # XXX: todo hardcoded, make it a config
                # XXX todo: make it select the stimulus which was cliekd and then check if a new dog appeared after that, not after last click.
                # todo: make better sounds (they should be shorter and start immediately)
                success_sound = True

            if success_sound:
                pygame.mixer.Sound("sound_success.mp3").play()
            else:
                pygame.mixer.Sound("sound_failure.mp3").play()

            click_times.append(current_time)

    if running and started and not paused:
        show_stimulus(current_image_i)
        current_image_i += 1

        pygame.time.delay(TEMPLATE["settings"]["TIME_OFF"] - int((time.time() - time_start_showing_gray_frame) * 1000))
    else: 
        clock.tick(GAME_FPS)
    
    current_time = time.time()
    if current_time - last_backup >= BACKUP_CONFIG_INTERVAL:
        save_events()
        last_backup = current_time
    
    if current_image_i == len(TEMPLATE["framedata"]):
        running = False
        save_events()

pygame.quit()