import time
import cv2
import os

from utils import roboflow_utils
import ui_constants
from utils import screen_utils
import environment


def test_screenshot():
    image = screen_utils.screenshot()
    cv2.imwrite("images/result_images/screen.png", image)


def test_crop_game():
    image = cv2.imread("ref_images/screenshot.png")
    crop = screen_utils.crop_area(image, ui_constants.game_top_left, ui_constants.game_bottom_right)
    cv2.imwrite("images/result_images/gameplay.png", crop)


def test_crop_timer():
    image = cv2.imread("images/result_images/gameplay.png")
    crop = screen_utils.crop_area(image, ui_constants.time_top_left, ui_constants.time_bottom_right)
    cv2.imwrite("images/result_images/timer.png", crop)


def text_read_text():
    image = cv2.imread("images/result_images/timer.png")
    text = screen_utils.read_text(image)
    if text == "2:44\n" or text == "2:44":
        print("success")
    minutes, seconds = text.split(":")
    timer = int(minutes) * 60 + int(seconds)
    print(f"Timer: {timer} seconds")


def get_images(n):
    image = screen_utils.screenshot()
    crop = screen_utils.crop_area(image, ui_constants.game_top_left, ui_constants.game_bottom_right)
    cv2.imwrite(f"images/real_ss/ss_{n}.png", crop)
    print("wrote image")


def take_ss():
    i = 0
    while True:
        time.sleep(3)
        get_images(i)
        i += 1


def get_detections(gameplay, client_in, n):
    elixir = environment.get_elixir()
    timer = environment.get_timer(gameplay)
    results = roboflow_utils.detect_troop(client_in, gameplay)["predictions"]

    rf_w = 640
    rf_h = 640

    img_h, img_w = gameplay.shape[:2]

    scale_x = img_w / rf_w
    scale_y = img_h / rf_h

    for det in results:
        x_center = det["x"] * scale_x
        y_center = det["y"] * scale_y
        w_box = det["width"] * scale_x
        h_box = det["height"] * scale_y

        # convert center → corners
        x1 = int(x_center - w_box / 2)
        y1 = int(y_center - h_box / 2)
        x2 = int(x_center + w_box / 2)
        y2 = int(y_center + h_box / 2)

        cv2.rectangle(gameplay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(gameplay, f"{det['class']}:{det['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    ok = cv2.imwrite(f"images/detections/{n}_{elixir}.png", gameplay)
    print(timer)


def crop_detections(gameplay, client_in, n):
    results = roboflow_utils.detect_troop(client_in, gameplay)["predictions"]

    rf_w = 640
    rf_h = 640

    img_h, img_w = gameplay.shape[:2]

    scale_x = img_w / rf_w
    scale_y = img_h / rf_h

    for index, det in enumerate(results):
        x_center = det["x"] * scale_x
        y_center = det["y"] * scale_y
        w_box = det["width"] * scale_x
        h_box = det["height"] * scale_y

        # convert center → corners
        x1 = int(x_center - w_box / 2)
        y1 = int(y_center - h_box / 2)
        x2 = int(x_center + w_box / 2)
        y2 = int(y_center + h_box / 2)

        cropped = screen_utils.crop_area(gameplay, (x1, y1), (x2, y2))
        cv2.imwrite(f"troops/{n}_{index}.png", cropped)


def label_images(mode="detect"):
    folder = "images/real_ss"
    client = roboflow_utils.init_roboflow()

    # allowed image extensions
    exts = (".png", ".jpg", ".jpeg")

    for idx, filename in enumerate(sorted(os.listdir(folder))):
        if filename.lower().endswith(exts):

            path = os.path.join(folder, filename)
            image = cv2.imread(path)

            if image is None:
                print(f"Failed to read image: {path}")
                continue

            print(f"Processing {filename}...")
            if mode == "detect":
                get_detections(image, client, idx)
            else:
                crop_detections(image, client, idx)

def winner_detection():
    cv2.imread("images/test_images/end_screen.png")
    win = screen_utils.crop_area(
        cv2.imread("images/test_images/end_screen.png"),
        ui_constants.victory_banner_top_left,
        ui_constants.victory_banner_bottom_right
    )
    lose = screen_utils.crop_area(
        cv2.imread("images/test_images/end_screen.png"),
        ui_constants.defeat_banner_top_left,
        ui_constants.defeat_banner_bottom_right
    )
    if(screen_utils.read_text(win) == "Winner!\n"):
        print("Victory detected")
    if(screen_utils.read_text(lose) == "Winner!\n"):
        print("Defeat detected")

if __name__ == "__main__":
    # label_images(mode="crop")
    text_read_text()
    

    