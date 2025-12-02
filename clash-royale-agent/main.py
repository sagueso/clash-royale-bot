import cv2
import supervision as sv

from utils import roboflow_utils, screen_utils
import environment
import time


if __name__ == '__main__':
    # Mainloop
    data = []
    try:

        client = roboflow_utils.init_roboflow()
        print("roboflow client initialized.")
        environment.start_battle()
        print('Press Ctrl-C to quit.')
        print("Starting battle")

        while True:
            initial_time = time.time()
            image = screen_utils.screenshot()
            gameplay = environment.get_game_screen(image)
            elixir = environment.get_elixir()
            timer = environment.get_timer(gameplay)
            results = roboflow_utils.detect_troop(client, gameplay)["predictions"]
            print(results)
            #detections = sv.Detections.from_inference(results["predictions"])
            #data.append([elixir, timer, detections.data["class_name"]])

            print("elapsed:", time.time() - initial_time)

    except KeyboardInterrupt:
        print('\n')
        for d in data:
            print(d)
