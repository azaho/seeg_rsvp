import os
import cv2
import numpy as np
import json
import math
import random

from config import *


def draw_cross(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    thickness = math.ceil(VISUAL_CROSS_THICKNESS * VISUAL_STIMULUS_WIDTH)
    size = math.ceil(VISUAL_CROSS_SIZE * VISUAL_STIMULUS_WIDTH)

    if VISUAL_CROSS_BACKGROUND:
        cv2.circle(img, center, size, VISUAL_CROSS_BACKGROUND_COLOR, cv2.FILLED)
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), VISUAL_CROSS_COLOR, thickness)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), VISUAL_CROSS_COLOR, thickness)
    return img

def draw_square(img: np.ndarray) -> np.ndarray:
    h, w = SCREEN_HEIGHT, SCREEN_WIDTH
    size = int(VISUAL_SQUARE_SIZE * SCREEN_HEIGHT)
    top = h - size
    right = w - size

    cv2.rectangle(img, (right, top), (right + size, top + size), VISUAL_SQUARE_COLOR, cv2.FILLED)
    return img

def get_gray_frame(cross: bool = True, square: bool = True):
    gray_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    x_offset = (SCREEN_WIDTH - VISUAL_STIMULUS_WIDTH) // 2
    y_offset = (SCREEN_HEIGHT - VISUAL_STIMULUS_HEIGHT) // 2

    gray_frame[y_offset:y_offset + VISUAL_STIMULUS_HEIGHT, x_offset:x_offset + VISUAL_STIMULUS_WIDTH] = VISUAL_GRAY_COLOR

    if cross:
        gray_frame = draw_cross(gray_frame)
    if square:
        gray_frame = draw_square(gray_frame)
    return gray_frame


def generate_random_crop(img: np.ndarray) -> np.ndarray:
    """Random square crop of `size`×`size` from a larger image, where size is the shorter side of the image."""
    h, w = img.shape[:2]
    size = min(h, w)

    top  = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return top, left, size, size

def resize_shorter_to(img: np.ndarray, target: int) -> np.ndarray:
    """Scale so the shorter side == target while preserving aspect ratio."""
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = target, int(w * target / h)
    else:
        new_w, new_h = target, int(h * target / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def crop_image(img: np.ndarray, crop_dict: dict) -> np.ndarray:
    """Crop image according to specified dimensions."""
    top = crop_dict["top"]
    left = crop_dict["left"] 
    width = crop_dict["width"]
    height = crop_dict["height"]
    return img[top:top + height, left:left + width]

def preprocess_image(framedata, cross: bool = True):
    img = cv2.imread(framedata["image_path"])
    img = crop_image(img, framedata["crop"])
    img = resize_shorter_to(img, VISUAL_STIMULUS_WIDTH)
    
    if cross:
        img = draw_cross(img)
    return img