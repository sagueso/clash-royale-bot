import cv2
import numpy as np
import pyautogui
import mss
import pytesseract


# https://www.youtube.com/watch?v=e3NeWoXTkuQ&ab_channel=AlanZheng


sct = mss.mss()


def screenshot():
    monitor = sct.monitors[1]  # full screen
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def find(image_path, screen):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[::-1]

    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screen_gray, image, cv2.TM_CCOEFF_NORMED)

    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(screen, pt, (pt[0] + height, pt[1] + width), (0, 0, 255), 2)
        return True, pt, (pt[0] + height, pt[1] + width)

    return False, None, None


def click(x_click, y_click):
    pyautogui.click(x=x_click, y=y_click)


def find_n_click(image_path, screen):
    is_found, top_left, bottom_right = find(image_path, screen)
    if is_found:
        print((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)
        click((top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2)

    return is_found


def click_n_click(initial_point, final_point):
    click(initial_point[0], initial_point[1])
    click(final_point[0], final_point[1])


def crop_area(screen, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right
    crop = screen[y1:y2, x1:x2]
    return crop


def read_text(image):
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(th, config="--psm 7")


def get_pixel_color(pixel):
    """Get RGB color at a specific pixel position.
    
    Args:
        pixel: Tuple of (x, y) coordinates
        
    Returns:
        Tuple of (r, g, b) values
    """
    r, g, b = pyautogui.pixel(pixel[0], pixel[1])
    return (r, g, b)


def check_pixel_color(pixel, target_color):
    tolerance = 80
    r, g, b = get_pixel_color(pixel)
    # pyautogui.moveTo(pixel[0], pixel[1])
    # print(f"{r}, {g}, {b}")
    return (abs(r - target_color[0]) <= tolerance) and (abs(g - target_color[1]) <= tolerance) \
        and (abs(b - target_color[2]) <= tolerance)
