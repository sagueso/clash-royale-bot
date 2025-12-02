import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import cv2

load_dotenv()


def init_roboflow():
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",  # use roboflow server
        api_key=os.getenv('ROBOFLOW_API_KEY')  # optional to access your private data and models
    )
    return client


def detect_troop(client, image):
    resized = cv2.resize(image, (640, 640))
    result = client.infer(resized, model_id="cr-troop-tower-side-detection-y0qpd/3")

    return result


