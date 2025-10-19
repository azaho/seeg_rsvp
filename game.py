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

### INSTRUCTION SCREEN ###

def show_instruction_screen():
    """Display instruction screen with gray square + cross and instructions text"""
    screen.blit(gray_frame_square, (0, 0))
    
    # Create font for instructions
    font = pygame.font.Font(None, 48)  # Default font, size 48
    
    # Instruction text
    instructions = [
        "A stream of images will keep rapidly flowing as you look at the cross",
        "in the middle of the screen. Whenever you see a dog in the image,",
        "press MOUSE CLICK. A session lasts up to 8 minutes.",
        "",
        "Click mouse or press any key to start."
    ]
    
    # Render and display text
    y_offset = SCREEN_HEIGHT // 6  # Start at 1/6 from top
    for line in instructions:
        # Render shadow text first
        shadow_surface = font.render(line, True, (0, 0, 0))  # Black text
        shadow_rect = shadow_surface.get_rect(center=(SCREEN_WIDTH // 2 + 2, y_offset + 2))
        screen.blit(shadow_surface, shadow_rect)
        
        # Render main text on top
        text_surface = font.render(line, True, (255, 255, 255))  # White text
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(text_surface, text_rect)
        y_offset += 60  # Space between lines
    pygame.display.flip()
    
    # Wait for mouse click or key press
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to quit
            elif event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                waiting = False
    
    # Show gray frame without square after instruction screen
    blit_gray_frame(square=False)
    create_event("instructions_completed")
    return True  # Signal to continue

# Show instruction screen
if not show_instruction_screen():
    pygame.quit()
    sys.exit()

### GAME LOOP ###

stream_running = False
game_running = True
current_image_i = 0

CLICK_TIMES = []
SUCCESS_CLICKS = []
FAILURE_CLICKS = []
STIMULUS_ONSET_TIMES_PREDETERMINED = np.zeros(len(TEMPLATE["framedata"]))
STIMULUS_OFFSET_TIMES_PREDETERMINED = np.zeros(len(TEMPLATE["framedata"]))
STIMULUS_ONSET_TIMES_ACTUAL = np.zeros(len(TEMPLATE["framedata"]))
STIMULUS_OFFSET_TIMES_ACTUAL = np.zeros(len(TEMPLATE["framedata"]))
target_stimulus_idx = [i for i, framedata in enumerate(TEMPLATE["framedata"]) if framedata["target"]]

def show_stimulus(image_i):
    blit_image(image_i)
    create_event("show_stimulus", image_i=image_i)
    STIMULUS_ONSET_TIMES_ACTUAL[image_i] = time.time()


def get_last_target_stimulus_idx(current_image_i):
    # Find the last target stimulus index that is <= current_image_i
    for idx in reversed(target_stimulus_idx):
        if idx <= current_image_i:
            return idx
    return None

def start_stream():
    global stream_running, current_image_i
    assert not stream_running, "Stream is already running"

    start_i = current_image_i
    
    start_time = time.time()
    stream_running = True
    for counter_i, image_i in enumerate(range(start_i, len(TEMPLATE["framedata"]))):
        STIMULUS_ONSET_TIMES_PREDETERMINED[image_i] = start_time + counter_i * (TEMPLATE["settings"]["TIME_ON"] + TEMPLATE["settings"]["TIME_OFF"]) / 1000
        STIMULUS_OFFSET_TIMES_PREDETERMINED[image_i] = start_time + counter_i * (TEMPLATE["settings"]["TIME_ON"] + TEMPLATE["settings"]["TIME_OFF"]) / 1000 + TEMPLATE["settings"]["TIME_ON"] / 1000
    create_event("stream_start", start_i=start_i)

def process_click():
    global current_image_i, target_stimulus_idx, targets_caught

    print(f"Mouse clicked at position: {pygame.mouse.get_pos()}")
    create_event("mouse_click")

    success_sound = False
    current_time = time.time()

    for t_idx in target_stimulus_idx:
        if t_idx > current_image_i:
            break
        if current_time - STIMULUS_ONSET_TIMES_ACTUAL[t_idx] > 2: # TODO: hardcoded 2 seconds
            continue
        if t_idx in targets_caught:
            continue
        targets_caught.add(t_idx)
        success_sound = True
        break

    if success_sound:
        pygame.mixer.Sound("sound_success.mp3").play()
        SUCCESS_CLICKS.append(current_time)
    else:
        pygame.mixer.Sound("sound_failure.mp3").play()
        FAILURE_CLICKS.append(current_time)
    CLICK_TIMES.append(current_time)

def pause():
    global stream_running
    assert stream_running, "Game is already paused"
    stream_running = False
    blit_gray_frame(square=False)
    create_event("pause")

def save_data():
    log_arrays = {
        "CLICK_TIMES": CLICK_TIMES,
        "SUCCESS_CLICKS": SUCCESS_CLICKS,
        "FAILURE_CLICKS": FAILURE_CLICKS,
        "STIMULUS_ONSET_TIMES_PREDETERMINED": STIMULUS_ONSET_TIMES_PREDETERMINED,
        "STIMULUS_OFFSET_TIMES_PREDETERMINED": STIMULUS_OFFSET_TIMES_PREDETERMINED,
        "STIMULUS_ONSET_TIMES_ACTUAL": STIMULUS_ONSET_TIMES_ACTUAL,
        "STIMULUS_OFFSET_TIMES_ACTUAL": STIMULUS_OFFSET_TIMES_ACTUAL,
        "TARGET_STIMULUS_IDX": target_stimulus_idx,
    }
    log_arrays = {k: np.array(v) for k, v in log_arrays.items()}
    np.savez(os.path.join(SAVE_DIR, "log_arrays.npz"), **log_arrays)

last_backup = time.time()
stimulus_onset_processed = set()
stimulus_offset_processed = set()
targets_caught = set()

while game_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                if not stream_running:
                    start_stream()
            elif event.key == pygame.K_p:
                if stream_running:
                    pause()
                else:
                    start_stream()
            elif event.key == pygame.K_SPACE:
                process_click()
            elif event.key == pygame.K_t:
                create_event("trigger")
            elif event.key == pygame.K_q:
                game_running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            process_click()

    current_time = time.time()

    if stream_running:
        if current_image_i not in stimulus_onset_processed and current_time >= STIMULUS_ONSET_TIMES_PREDETERMINED[current_image_i]:
            show_stimulus(current_image_i)
            stimulus_onset_processed.add(current_image_i)

        if current_image_i not in stimulus_offset_processed and current_time >= STIMULUS_OFFSET_TIMES_PREDETERMINED[current_image_i]:
            blit_gray_frame(square=False)
            stimulus_offset_processed.add(current_image_i)
            current_image_i += 1

        if current_image_i >= len(TEMPLATE["framedata"]):
            stream_running = False
            create_event("stream_end")
    
    if current_time - last_backup >= BACKUP_CONFIG_INTERVAL:
        save_events()
        last_backup = current_time

save_events()
save_data()
pygame.quit()