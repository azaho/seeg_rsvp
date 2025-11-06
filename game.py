import json
import pygame
import os
import screen_setup
from config import *
import time
import shutil
import sys
import cv2
from tqdm import tqdm

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

SAVE_DIR = f"trial_data/trial_data_{get_timecode()}/"
os.makedirs(SAVE_DIR, exist_ok=True)

shutil.copy2('config.py', os.path.join(SAVE_DIR, f'config_{get_timecode()}.py'))
shutil.copy2(template_path, os.path.join(SAVE_DIR, f'template.json'))
# Copy the whole parent directory of template_path (i.e., the template folder) to SAVE_DIR/templates/{template_name}
template_parent_dir = os.path.dirname(template_path)
template_name = os.path.basename(template_parent_dir)
shutil.copytree(template_parent_dir, os.path.join(SAVE_DIR, template_name), dirs_exist_ok=True)


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
print("\nFound displays:")
for i, (width, height) in enumerate(desktop_sizes):
    print(f"Display {i}: {width}x{height}")
print(f"Using display {SCREEN_ID}\n")


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
for framedata in tqdm(TEMPLATE["framedata"], desc="Preparing frames"):
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
        "GAME RULES:",
        "1. Keep your eyes looking at the crosshair in the middle of the screen.",
        "2. Whenever you see a dog, press MOUSE CLICK!",
        "3. You have 2 seconds to catch each dog.",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "A session lasts up to 8 minutes.",
        "Click mouse or press any key to start the session."
    ]
    
    # Render and display text
    y_offset = SCREEN_HEIGHT // 9  # Start at 1/9 from top
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

### RESULTS SCREEN ###

def show_results_screen(show_targets_caught=True, show_targets_missed=False, show_wrong_clicks=True, show_percentage=False, show_medal=True):
    """Display results screen with performance metrics and medals.
    show_targets_caught: show number of targets caught
    show_targets_missed: show number of targets missed
    show_wrong_clicks: show number of wrong clicks
    show_percentage: show percentage of targets caught
    show_medal: show medal
    """
    screen.fill((0, 0, 0))
    
    # Calculate metrics
    num_targets = len(target_stimulus_idx)
    num_targets_caught = len(targets_caught)
    num_targets_missed = num_targets - num_targets_caught
    num_wrong_clicks = len(FAILURE_CLICKS)
    
    # Calculate percentage
    percentage = (num_targets_caught / num_targets * 100) if num_targets > 0 else 0
    
    # Determine medal
    medal = ""
    medal_color = (255, 255, 255)  # White default
    if percentage >= 90:
        medal = "GOLD MEDAL!"
        medal_color = (255, 215, 0)  # Gold
    elif percentage >= 80:
        medal = "SILVER MEDAL!"
        medal_color = (192, 192, 192)  # Silver
    elif percentage >= 50:
        medal = "BRONZE MEDAL!"
        medal_color = (205, 127, 50)  # Bronze
    
    # Create fonts
    title_font = pygame.font.Font(None, 72)
    font_large = pygame.font.Font(None, 60)
    font = pygame.font.Font(None, 48)
    
    # Results text
    y_offset = SCREEN_HEIGHT // 6
    
    # Title
    title_text = "Results"
    title_surface = title_font.render(title_text, True, (255, 255, 255))
    title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
    screen.blit(title_surface, title_rect)
    y_offset += 100
    
    if show_targets_caught:
        # Targets caught
        caught_text = f"Targets caught: {num_targets_caught} / {num_targets}"
        caught_surface = font_large.render(caught_text, True, (255, 255, 255))
        caught_rect = caught_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(caught_surface, caught_rect)
        y_offset += 80
    
    if show_targets_missed:
        # Targets missed
        missed_text = f"Targets missed: {num_targets_missed}"
        missed_surface = font.render(missed_text, True, (255, 255, 255))
        missed_rect = missed_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(missed_surface, missed_rect)
        y_offset += 70
    
    if show_wrong_clicks:
        # Wrong clicks
        wrong_text = f"Wrong clicks: {num_wrong_clicks}"
        wrong_surface = font.render(wrong_text, True, (255, 255, 255))
        wrong_rect = wrong_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(wrong_surface, wrong_rect)
        y_offset += 80
    
    if show_percentage:
        # Percentage
        percentage_text = f"Accuracy: {percentage:.1f}%"
        percentage_surface = font_large.render(percentage_text, True, (255, 255, 255))
        percentage_rect = percentage_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(percentage_surface, percentage_rect)
        y_offset += 90
    

    if "tutorial" in template_name:
        show_medal = False # don't show medal for tutorial session

    # Medal (if earned)
    if show_medal and medal:
        # Draw medal icon (circle with inner circle)
        medal_icon_radius = 40
        medal_icon_x = SCREEN_WIDTH // 2 - 200
        medal_icon_y = y_offset
        
        # Outer circle (darker shade)
        outer_color = tuple(int(c * 0.7) for c in medal_color)
        pygame.draw.circle(screen, outer_color, (medal_icon_x, medal_icon_y), medal_icon_radius)
        
        # Inner circle (main color)
        pygame.draw.circle(screen, medal_color, (medal_icon_x, medal_icon_y), medal_icon_radius - 5)
        
        # Inner shine circle (lighter shade)
        shine_color = tuple(min(int(c * 1.3), 255) for c in medal_color)
        pygame.draw.circle(screen, shine_color, (medal_icon_x - 8, medal_icon_y - 8), 12)
        
        # Medal text
        medal_surface = font_large.render(medal, True, medal_color)
        medal_rect = medal_surface.get_rect(center=(SCREEN_WIDTH // 2 + 50, y_offset))
        screen.blit(medal_surface, medal_rect)
        y_offset += 90
    
    y_offset += 100 + 20

    if "tutorial" not in template_name: # don't show thank you message for tutorial session
        # Thank you message
        thank_you_text = "thank you for contributing to science!"
        thank_you_surface = font.render(thank_you_text, True, (180, 220, 180))
        thank_you_rect = thank_you_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        screen.blit(thank_you_surface, thank_you_rect)
        y_offset += 60

    # Exit instruction
    exit_text = "Press Q to complete the session"
    exit_surface = font.render(exit_text, True, (200, 200, 200))
    exit_rect = exit_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
    screen.blit(exit_surface, exit_rect)
    
    pygame.display.flip()
    
    # Wait for Q key to quit
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    waiting = False
    
    create_event("results_screen_completed", 
                 targets_caught=num_targets_caught,
                 targets_missed=num_targets_missed,
                 wrong_clicks=num_wrong_clicks,
                 percentage=percentage,
                 medal=medal)

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
STIMULUS_ONSET_TIMES_PREDETERMINED = None
STIMULUS_OFFSET_TIMES_PREDETERMINED = None
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
    global stream_running, current_image_i, STIMULUS_ONSET_TIMES_PREDETERMINED, STIMULUS_OFFSET_TIMES_PREDETERMINED
    assert not stream_running, "Stream is already running"

    start_i = current_image_i
    start_time = time.time()
    stream_running = True

    if STIMULUS_ONSET_TIMES_PREDETERMINED is None:
        assert current_image_i == 0, "Current image index must be 0 if stimulus onset times are not provided in the template"
        STIMULUS_ONSET_TIMES_PREDETERMINED = np.array(TEMPLATE["settings"]["STIMULUS_ONSET_TIMES_PREDETERMINED"]) + start_time
        STIMULUS_OFFSET_TIMES_PREDETERMINED = np.array(TEMPLATE["settings"]["STIMULUS_OFFSET_TIMES_PREDETERMINED"]) + start_time
    else:
        STIMULUS_ONSET_TIMES_PREDETERMINED[current_image_i:] = np.array(TEMPLATE["settings"]["STIMULUS_ONSET_TIMES_PREDETERMINED"][current_image_i:]) + start_time
        STIMULUS_OFFSET_TIMES_PREDETERMINED[current_image_i:] = np.array(TEMPLATE["settings"]["STIMULUS_OFFSET_TIMES_PREDETERMINED"][current_image_i:]) + start_time
    
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
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in [1, 3]:  # Left or right click only
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
            # Show results screen after stream ends
            show_results_screen()
            game_running = False  # End the game after showing results
    
    if current_time - last_backup >= BACKUP_CONFIG_INTERVAL:
        save_events()
        last_backup = current_time

        # log progress
        session_progress = (current_image_i / len(TEMPLATE["framedata"])) * 100
        print(f"Session progress: {session_progress:.1f}%")

save_events()
save_data()
pygame.quit()