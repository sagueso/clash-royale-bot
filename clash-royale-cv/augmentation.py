import albumentations as A
import cv2
import json
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "./dataset/train"
INPUT_JSON = "./dataset/train/_annotations.coco.json"
OUTPUT_DIR = "./datasets/v_2/train"
OUTPUT_JSON = f"{OUTPUT_DIR}/_annotations.coco.json"
NUM_AUGMENTATIONS_PER_IMAGE = 2  # How many new versions to create per image

# --- AUGMENTATIONS ---
transform = A.Compose([
    #A.RandomScale(scale_limit=0.3, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50), p=1),
        A.ISONoise(p=1),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=1),
    ], p=0.4),
    A.OneOf([
        A.MotionBlur(p=1),
        A.GaussianBlur(blur_limit=3, p=1),
    ], p=0.2),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.HorizontalFlip(p=0.5),
    
    # Pad if scaling made it small, crop if large
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=640, width=640) 
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# --- PROCESSING SCRIPT ---
def augment_coco():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    with open(INPUT_JSON, 'r') as f:
        coco_data = json.load(f)

    new_images = []
    new_annotations = []
    ann_id_counter = 1000000  # Start high to avoid ID conflicts
    img_id_counter = 1000000

    # Group annotations by image_id
    img_to_anns = {img['id']: [] for img in coco_data['images']}
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    print("Augmenting images...")
    for img_info in tqdm(coco_data['images']):
        # 1. Copy original image first
        original_path = os.path.join(INPUT_DIR, img_info['file_name'])
        shutil.copy(original_path, os.path.join(OUTPUT_DIR, img_info['file_name']))
        new_images.append(img_info)
        new_annotations.extend(img_to_anns[img_info['id']])

        # 2. Read Image
        image = cv2.imread(original_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = img_to_anns[img_info['id']]
        
        bboxes = [a['bbox'] + [a['category_id']] for a in anns] # Format for Albumentations
        category_ids = [a['category_id'] for a in anns]

        # 3. Generate Augmented Versions
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            try:
                # Apply transform
                transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                trans_img = transformed['image']
                trans_bboxes = transformed['bboxes']

                # Save Image
                new_filename = f"aug_{i}_{img_info['file_name']}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, new_filename), cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR))

                # Create New Image Info
                new_img_id = img_id_counter
                img_id_counter += 1
                new_images.append({
                    "id": new_img_id,
                    "file_name": new_filename,
                    "height": trans_img.shape[0],
                    "width": trans_img.shape[1]
                })

                # Create New Annotations
                for box, cat_id in zip(trans_bboxes, category_ids):
                    # Remove last element (category_id) from box list for COCO format
                    clean_box = list(box[:4]) 
                    new_annotations.append({
                        "id": ann_id_counter,
                        "image_id": new_img_id,
                        "category_id": cat_id,
                        "bbox": clean_box,
                        "area": clean_box[2] * clean_box[3],
                        "iscrowd": 0,
                        "segmentation": [] # Bounding box only
                    })
                    ann_id_counter += 1
            except Exception as e:
                print(f"Skipping aug for {img_info['file_name']}: {e}")

    # Save final JSON
    output_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data['categories']
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_coco, f)
    print(f"Done! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    augment_coco()