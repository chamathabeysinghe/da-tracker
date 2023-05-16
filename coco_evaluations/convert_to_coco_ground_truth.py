from dataset.configuration import VIDEO_CLIPS_TARGET
import pandas as pd
import json
import os
import cv2
DATASET_DIR = '/Users/cabe0006/Projects/monash/cvpr_data'

SCALE = 4.0
split = 'test'
file_names = VIDEO_CLIPS_TARGET['val'] + VIDEO_CLIPS_TARGET['test']
json_obj = {
    "categories": [
        {
            "id": 0,
            "name": "ant"
        }
    ],
    "images": [],
    "annotations": []
}
image_count = 0
detection_count = 0
COCO_DIR = '/Users/cabe0006/Projects/monash/trackformer/coco_evaluations/raw_data'
for file in ['CU30L1B6In_0']:
    df = pd.read_csv(os.path.join(DATASET_DIR, 'raw_data', 'csv', f'{file}.csv'))
    num_frames = max(df.image_id.unique()) + 1
    # frames = get_frames(os.path.join(DATASET_DIR, 'raw_data', 'videos', f'{file}.mp4'), max=num_frames)
    # image_dir = os.path.join(DATASET_DIR, 'detection_dataset_nature', split)
    # os.makedirs(image_dir, exist_ok=True)
    for image_id in range(num_frames):
        image_count += 1
        image_name = f'{file}_{image_id:06d}'
        # cv2.imwrite(os.path.join(image_dir, f'{image_name}.jpg'), frames[image_id])
        json_obj["images"].append({
            "id": image_count,
            "license": 1,
            "file_name": "{}.jpg".format(image_name),
            "height": int(2168.0 / SCALE),
            "width": int(4096.0 / SCALE),
            "date_captured": "null"
        })
        for index, row in df.loc[df['image_id'] == image_id].iterrows():
            detection_count += 1
            ant_details = {
                "id": detection_count,
                "image_id": image_count,
                "category_id": 0,
                "bbox": [int(row["x"] / SCALE), int(row["y"] / SCALE), int(row["w"] / SCALE), int(row["h"] / SCALE)],
                "area": int(row["w"] * row["h"]),
                "iscrowd": 0
            }
            json_obj["annotations"].append(ant_details)
    with open(os.path.join(COCO_DIR, f'ground-truth-{file}.json'), 'w') as outfile:
        json.dump(json_obj, outfile)

