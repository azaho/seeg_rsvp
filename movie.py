import json
import pygame
import os
from config import *
import time
import sys
import cv2
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Play a video file with visual timing squares')
parser.add_argument('movie_path', help='Path to the MP4 video file')
parser.add_argument('--square-interval', '-s', type=int, default=100,
                   help='Show white square every N frames (default: 100)')
parser.add_argument('--start-time', '-t', default=None,
                   help='Start time in HH:MM:SS format (default: 0)')

# Parse arguments
args = parser.parse_args()

# Validate movie path exists
movie_path = args.movie_path
assert os.path.exists(movie_path), f"Movie file {movie_path} not found."

# Get square interval and start time
SQUARE_INTERVAL = args.square_interval
START_TIME_STR = args.start_time

def parse_timestamp(time_str):
    """Convert HH:MM:SS to seconds"""
    if not time_str:
        return 0
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])

START_TIME_SECONDS = parse_timestamp(START_TIME_STR) if START_TIME_STR else 0

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
create_event("application_start", movie_path=movie_path, square_interval=SQUARE_INTERVAL, start_time_seconds=START_TIME_SECONDS)

### SETTING UP THE SAVE DIRECTORY ###

def get_timecode():
    return time.strftime('%Y-%m-%d_%H-%M-%S')

SAVE_DIR = f"trial_data/{get_timecode()}/"
os.makedirs(SAVE_DIR, exist_ok=True)

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
pygame.mixer.init()

# Setup font for time display
pygame.font.init()
time_font = pygame.font.Font(None, 48)  # Default font, size 48

desktop_sizes = pygame.display.get_desktop_sizes()
if SCREEN_ID < 0: SCREEN_ID = len(desktop_sizes) - 1
print("\nFound displays:")
for i, (width, height) in enumerate(desktop_sizes):
    print(f"Display {i}: {width}x{height}")
print(f"Using display {SCREEN_ID}\n")

monitor_width, monitor_height = desktop_sizes[SCREEN_ID]
screen = pygame.display.set_mode((monitor_width, monitor_height), pygame.NOFRAME | pygame.FULLSCREEN, display=SCREEN_ID)
pygame.display.set_caption("Movie Player")
pygame.display.set_allow_screensaver(False)  # Prevent screensaver from activating
pygame.event.set_grab(True)  # Keep input focus
pygame.mouse.set_visible(False)  # Hide the cursor

print(f"Screen resolution: {monitor_width}x{monitor_height}")

### LOADING VIDEO ###

print(f"Loading video: {movie_path}")
video = cv2.VideoCapture(movie_path)

if not video.isOpened():
    print(f"Error: Could not open video file {movie_path}")
    pygame.quit()
    sys.exit(1)

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info: {video_width}x{video_height}, {fps} FPS, {total_frames} frames")
print(f"Square will appear every {SQUARE_INTERVAL} frames")
print(f"Duration: {total_frames/fps:.2f} seconds")

# Seek to start time if specified
if START_TIME_SECONDS > 0:
    print(f"Seeking to start time: {START_TIME_STR} ({START_TIME_SECONDS:.2f} seconds)")
    video.set(cv2.CAP_PROP_POS_MSEC, START_TIME_SECONDS * 1000)
    start_frame = int(START_TIME_SECONDS * fps)
    print(f"Starting from frame {start_frame}")

# Load audio from raw_mp3 subfolder
audio_available = False
try:
    # Get the directory and base name of the movie file
    movie_dir = os.path.dirname(movie_path)
    movie_basename = os.path.splitext(os.path.basename(movie_path))[0]
    
    # Construct path to audio file in raw_mp3 subfolder
    audio_path = os.path.join(movie_dir, 'raw_mp3', f'{movie_basename}.mp3')
    
    if os.path.exists(audio_path):
        pygame.mixer.music.load(audio_path)
        print(f"Audio loaded from: {audio_path}")
        
        # Pre-warm the audio by starting playback and immediately pausing
        # This forces pygame to decode and prepare the audio stream, preventing lag on first play
        print("Preparing audio stream...")
        pygame.mixer.music.play(start=START_TIME_SECONDS)
        pygame.mixer.music.pause()
        print("Audio ready")
        
        audio_available = True
    else:
        print(f"Warning: Audio file not found at: {audio_path}")
except Exception as e:
    print(f"Warning: Could not load audio: {e}")
    audio_available = False

def to_pygame_frame(frame):
    """Convert OpenCV frame to pygame surface"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Scale frame to fill screen while maintaining aspect ratio
    frame_aspect = video_width / video_height
    screen_aspect = monitor_width / monitor_height
    
    if frame_aspect > screen_aspect:
        # Video is wider than screen
        new_width = monitor_width
        new_height = int(monitor_width / frame_aspect)
    else:
        # Video is taller than screen
        new_height = monitor_height
        new_width = int(monitor_height * frame_aspect)
    
    scaled_frame = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return pygame.image.frombuffer(scaled_frame.tobytes(), (new_width, new_height), 'RGB')

def draw_white_square():
    """Draw white square in bottom-right corner"""
    size = int(VISUAL_SQUARE_SIZE * monitor_height)
    top = monitor_height - size
    right = monitor_width - size
    pygame.draw.rect(screen, VISUAL_SQUARE_COLOR, (right, top, size, size))

def draw_black_square():
    """Draw black square in bottom-right corner"""
    size = int(VISUAL_SQUARE_SIZE * monitor_height * 1.5) # make it 1.5x the size of the white square
    top = monitor_height - size
    right = monitor_width - size
    pygame.draw.rect(screen, (0, 0, 0), (right, top, size, size))

def format_time_hms(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def draw_time_display(current_seconds, size_multiplier=0.8):
    """Draw current time in HH:MM:SS format above the black square"""
    time_str = format_time_hms(current_seconds)
    
    # Scale the font size relative to screen height
    font_size = int(monitor_height * VISUAL_SQUARE_SIZE * size_multiplier * 0.25)
    time_font = pygame.font.Font(None, font_size)
    text_surface = time_font.render(time_str, True, (255, 255, 255))  # White text
    
    # Get size of black square for positioning
    square_size = int(VISUAL_SQUARE_SIZE * monitor_height * 1.5)
    square_top = monitor_height - square_size
    square_right = monitor_width - square_size
    
    # Position text at bottom right of screen, just above black square
    text_rect = text_surface.get_rect()
    text_rect.right = monitor_width  # Align to right edge of screen
    text_rect.bottom = square_top  # Place directly above black square
    
    screen.blit(text_surface, text_rect)

### MOVIE PLAYBACK LOOP ###

print("\nControls:")
print("  SPACE or P: Play/Pause")
print("  S: Start playback")
print("  T: Trigger event")
print("  Mouse Click: Log click event")
print("  Q or ESC: Quit")
print()

running = True
playing = False
frame_count = int(START_TIME_SECONDS * fps) if START_TIME_SECONDS > 0 else 0
playback_start_time = None
next_frame_time = None
audio_started = False

# Fill screen with black initially
screen.fill((0, 0, 0))
pygame.display.flip()

def start_playback():
    """Start playback from initial state - audio will start when first frame is displayed"""
    global playback_start_time, next_frame_time, audio_started
    playback_start_time = time.time()
    next_frame_time = playback_start_time
    audio_started = False  # Will start audio when first frame is displayed

def resume_playback():
    """Resume playback from paused state"""
    global next_frame_time
    next_frame_time = time.time()
    if audio_available and audio_started:
        pygame.mixer.music.unpause()

def pause_playback():
    """Pause playback"""
    if audio_available and audio_started:
        pygame.mixer.music.pause()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Play/Pause toggle (SPACE or P keys)
            if event.key in [pygame.K_SPACE, pygame.K_p]:
                playing = not playing
                if playing:
                    if playback_start_time is None:
                        start_playback()
                        create_event("playback_start")
                        print("Playback started")
                    else:
                        resume_playback()
                        create_event("playback_resume")
                        print("Playback resumed")
                else:
                    pause_playback()
                    create_event("playback_pause", frame=frame_count)
                    print(f"Playback paused at frame {frame_count}")
            # Start playback with 's' key (only starts, doesn't pause)
            elif event.key == pygame.K_s:
                if not playing:
                    playing = True
                    if playback_start_time is None:
                        start_playback()
                    else:
                        resume_playback()
                    create_event("playback_start")
                    print("Playback started (S key)")
            # Trigger event
            elif event.key == pygame.K_t:
                create_event("trigger", frame=frame_count)
                print(f"Trigger event at frame {frame_count}")
            # Quit
            elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in [1, 3]:  # Left or right click only
            create_event("mouse_click", frame=frame_count, position=pygame.mouse.get_pos())
            print(f"Mouse clicked at frame {frame_count}, position: {pygame.mouse.get_pos()}")
    
    current_time = time.time()
    
    if playing:
        # Check if it's time to show the next frame
        if current_time >= next_frame_time:
            # Read next frame
            ret, frame = video.read()
            
            if ret:
                # Convert and display frame
                pygame_frame = to_pygame_frame(frame)
                
                # Center the frame on screen (in case it doesn't fill the whole screen)
                frame_rect = pygame_frame.get_rect()
                frame_rect.center = (monitor_width // 2, monitor_height // 2)
                
                screen.fill((0, 0, 0))  # Fill with black first
                screen.blit(pygame_frame, frame_rect)
                
                # Draw current time above the black square
                current_video_time = frame_count / fps
                draw_time_display(current_video_time)
                
                draw_black_square()
                # Draw white square every N frames
                if frame_count % SQUARE_INTERVAL == 0:
                    draw_white_square()
                    create_event("square_shown", frame=frame_count, movie_time=current_video_time)
                
                pygame.display.flip()
                
                # Start audio after first frame is displayed to ensure sync
                if not audio_started and audio_available:
                    pygame.mixer.music.unpause()
                    audio_started = True
                
                frame_count += 1
                
                # Calculate next frame time based on FPS
                frame_interval = 1.0 / fps
                next_frame_time += frame_interval
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Frame {frame_count}/{total_frames}; Current movie time: {format_time_hms(current_video_time)} ({progress:.1f}%)")
            else:
                # Video ended
                print(f"\nVideo playback complete ({frame_count} frames)")
                create_event("playback_complete", total_frames=frame_count)
                playing = False
                audio_started = False
                if audio_available:
                    pygame.mixer.music.stop()
                
                # Option to replay
                print("Press SPACE to replay or Q to quit")
        
        # Small sleep to avoid busy-waiting
        # time.sleep(0.001)
    else:
        # Small sleep when paused to avoid busy-waiting
        time.sleep(0.01)

# Cleanup
if audio_available:
    pygame.mixer.music.stop()
video.release()
save_events()
pygame.quit()
print(f"\nEvents saved to {SAVE_DIR}")