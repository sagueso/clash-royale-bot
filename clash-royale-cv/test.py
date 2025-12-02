import os
import cv2
import matplotlib.pyplot as plt
from rfdetr import RFDETRNano

def main(argv):
    weights_path = "./models/rfdetr_nano_v1/checkpoint_best_ema.pt"

    test_dir = f"./datasets/v_1/test/" 

    while os.path.isdir(f"./datasets/v_{data_set_version+1}/test/"):
        data_set_version += 1
        test_dir = f"./datasets/v_{data_set_version}/test/"
    
    for i in range(len(argv)):
        if argv[i].startswith("--model"):
            weights_path = argv[i + 1]
        if argv[i].startswith("--version"):
            data_set_version = argv[i + 1]
        if argv[i].startswith("--test_dir"):
            test_dir = argv[i + 1]

    images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]
    model = RFDETRNano(weights_path=weights_path)

    for img_file in images[:3]:
        img_path = os.path.join(test_dir, img_file)
        predictions = model.predict(source=img_path, conf=0.5) 
        for result in predictions:
            annotated_frame = result.plot() 
            
            # Display using Matplotlib (because cv2.imshow often freezes in notebooks)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()