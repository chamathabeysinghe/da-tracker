# from utils.configuration import DATASET_DIR
from dataset.configuration import VIDEO_CLIPS_TARGET
# from utils.video_tools import get_frames
import pandas as pd
import json
import os
import cv2


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
# CSV_DIR = '/Users/cabe0006/Projects/monash/trackformer/coco_evaluations/predicted_files'
CSV_DIR = '/Users/cabe0006/Projects/monash/trackformer/coco_evaluations/exp11d/csvs'
COCO_DIR = '/Users/cabe0006/Projects/monash/trackformer/coco_evaluations/exp11d/predicted_files'
# CU15L1B1In_0', 'CU20L1B1Out_0', 'CU15L1B4In_0', 'CU20L1B4Out_0', 'CU30L1B6Out_0', 'CU10L1B6In_0'

for file in ['CU30L1B6Out_0']:
    df = pd.read_csv(os.path.join(CSV_DIR, f'{file}'), names=['image_id', 'track_id', 'x', 'y', 'w', 'h', 'a', 'b', 'c', 'd'], header=None)
    # df = pd.read_csv('/Users/cabe0006/Projects/monash/cvpr_data/raw_data/csv/CU15L1B1In_0.csv')
    image_ids = list(map(lambda x: int(x), df.image_id.unique()))
    num_frames = max(image_ids) + 1
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
        for index, row in df.loc[df['image_id'] == image_id+1].iterrows():
            detection_count += 1
            ant_details = {
                # "id": detection_count,
                "image_id": image_count,
                "category_id": 0,
                "bbox": [row["x"], row["y"], row["w"], row["h"]],
                # "bbox": [float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
                # "bbox": [row["x"] / SCALE, row["y"] / SCALE, row["w"] / SCALE, row["h"] / SCALE],
                "score": 0.5
                # "area": int(row["w"] * row["h"]),
                # "iscrowd": 0
            }
            json_obj["annotations"].append(ant_details)
with open(os.path.join(COCO_DIR, f'test-predictions-{file}.json'), 'w') as outfile:
    json.dump(json_obj["annotations"], outfile)

