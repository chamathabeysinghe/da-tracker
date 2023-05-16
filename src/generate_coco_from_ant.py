"""
Generates COCO data and annotation structure from MOTChallenge data.
"""
import argparse
import configparser
import csv
import json
import os
import shutil

import numpy as np
import pycocotools.mask as rletools
import skimage.io as io
import torch
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_iou
import shutil

from trackformer.datasets.tracking.mots20_sequence import load_mots_gt

DATA_ROOT = 'DATA_ROOT_PATH'
MOTS_ROOT = 'MOT_PATH'
VIS_THRESHOLD = 0.0

ANT_SEQS_INFO = {
    # 'OU10B2L2Out_0': {'img_width': 1024, 'img_height': 542, 'seq_length': 720},
    # 'OU50B2L2Out_0': {'img_width': 1024, 'img_height': 542, 'seq_length': 720},
    # 'CU10L1B6Out_0': {'img_width': 1024, 'img_height': 542, 'seq_length': 720},
    # 'CU50L1B6Out_0': {'img_width': 1024, 'img_height': 542, 'seq_length': 500}
    }


def generate_coco_from_mot(split_name='train', seqs_names=None,
                           root_split='train', mots=False, mots_vis=False,
                           frame_range=None):
    """
    Generates COCO data from MOT.
    """
    global DATA_ROOT

    if frame_range is None:
        frame_range = {'start': 0.0, 'end': 1.0}

    if mots:
        DATA_ROOT = MOTS_ROOT
    root_split_path = os.path.join(DATA_ROOT, root_split)
    root_split_mots_path = os.path.join(MOTS_ROOT, root_split)
    coco_dir = os.path.join(DATA_ROOT, split_name)

    if os.path.isdir(coco_dir):
        shutil.rmtree(coco_dir)

    os.mkdir(coco_dir)

    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "ant",
                                  "name": "ant",
                                  "id": 1}]
    annotations['annotations'] = []

    annotations_dir = os.path.join(os.path.join(DATA_ROOT, 'annotations'))
    if not os.path.isdir(annotations_dir):
        os.mkdir(annotations_dir)
    annotation_file = os.path.join(annotations_dir, f'{split_name}.json')

    # IMAGE FILES
    img_id = 0

    seqs = sorted(os.listdir(root_split_path))

    if seqs_names is not None:
        seqs = [s for s in seqs if s in seqs_names]
    annotations['sequences'] = seqs
    annotations['frame_range'] = frame_range
    print(split_name, seqs)

    for seq in seqs:
        # CONFIG FILE
        config = configparser.ConfigParser()
        config_file = os.path.join(root_split_path, seq, 'seqinfo.ini')

        if os.path.isfile(config_file):
            config.read(config_file)
            img_width = int(config['Sequence']['imWidth'])
            img_height = int(config['Sequence']['imHeight'])
            seq_length = int(config['Sequence']['seqLength'])
        else:
            img_width = ANT_SEQS_INFO[seq]['img_width']
            img_height = ANT_SEQS_INFO[seq]['img_height']
            seq_length = ANT_SEQS_INFO[seq]['seq_length']

        seg_list_dir = os.listdir(os.path.join(root_split_path, seq, 'img1'))
        start_frame = int(frame_range['start'] * seq_length)
        end_frame = int(frame_range['end'] * seq_length)
        seg_list_dir = seg_list_dir[start_frame: end_frame]

        print(f"{seq}: {len(seg_list_dir)}/{seq_length}")
        seq_length = len(seg_list_dir)

        for i, img in enumerate(sorted(seg_list_dir)):

            if i == 0:
                first_frame_image_id = img_id

            annotations['images'].append({"file_name": f"{seq}_{img}",
                                          "height": img_height,
                                          "width": img_width,
                                          "id": img_id,
                                          "frame_id": i,
                                          "seq_length": seq_length,
                                          "first_frame_image_id": first_frame_image_id})

            img_id += 1
            shutil.copyfile(os.path.join(os.getcwd(), root_split_path, seq, 'img1', img), os.path.join(coco_dir, f"{seq}_{img}"))
            # os.symlink(os.path.join(os.getcwd(), root_split_path, seq, 'img1', img),
            #            os.path.join(coco_dir, f"{seq}_{img}"))

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        img_dict['file_name']: img_dict['id']
        for img_dict in annotations['images']}
    for seq in seqs:
        # GT FILE
        gt_file_path = os.path.join(root_split_path, seq, 'gt', 'gt.txt')
        if mots:
            gt_file_path = os.path.join(
                root_split_mots_path,
                seq.replace('MOT17', 'MOTS20'),
                'gt',
                'gt.txt')
        if not os.path.isfile(gt_file_path):
            continue

        seq_annotations = []
        if mots:
            mask_objects_per_frame = load_mots_gt(gt_file_path)
            for frame_id, mask_objects in mask_objects_per_frame.items():
                for mask_object in mask_objects:
                    # class_id = 1 is car
                    # class_id = 2 is pedestrian
                    # class_id = 10 IGNORE
                    if mask_object.class_id == 1:
                        continue

                    bbox = rletools.toBbox(mask_object.mask)
                    bbox = [int(c) for c in bbox]
                    area = bbox[2] * bbox[3]
                    image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.jpg", None)
                    if image_id is None:
                        continue

                    segmentation = {
                        'size': mask_object.mask['size'],
                        'counts': mask_object.mask['counts'].decode(encoding='UTF-8')}

                    annotation = {
                        "id": annotation_id,
                        "bbox": bbox,
                        "image_id": image_id,
                        "segmentation": segmentation,
                        "ignore": mask_object.class_id == 10,
                        "visibility": 1.0,
                        "area": area,
                        "iscrowd": 0,
                        "seq": seq,
                        "category_id": annotations['categories'][0]['id'],
                        "track_id": mask_object.track_id}

                    seq_annotations.append(annotation)
                    annotation_id += 1

            annotations['annotations'].extend(seq_annotations)
        else:

            seq_annotations_per_frame = {}
            with open(gt_file_path, "r") as gt_file:
                reader = csv.reader(gt_file, delimiter=' ' if mots else ',')

                for row in reader:
                    if int(float(row[6])) == 1 and (seq in ANT_SEQS_INFO or int(float(row[7])) == 1):
                        bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                        bbox = [int(c) for c in bbox]

                        area = bbox[2] * bbox[3]
                        visibility = float(row[8])
                        frame_id = int(float(row[0]))
                        image_id = img_file_name_to_id.get(f"{seq}_{frame_id:06d}.jpg", None)
                        if image_id is None:
                            continue
                        track_id = int(float(row[1]))
                    
                        annotation = {
                            "id": annotation_id,
                            "bbox": bbox,
                            "image_id": image_id,
                            "segmentation": [],
                            "ignore": 0 if visibility > VIS_THRESHOLD else 1,
                            "visibility": visibility,
                            "area": area,
                            "iscrowd": 0,
                            "seq": seq,
                            "category_id": annotations['categories'][0]['id'],
                            "track_id": track_id}

                        seq_annotations.append(annotation)
                        if frame_id not in seq_annotations_per_frame:
                            seq_annotations_per_frame[frame_id] = []
                        seq_annotations_per_frame[frame_id].append(annotation)

                        annotation_id += 1

            annotations['annotations'].extend(seq_annotations)

            #change ignore based on MOTS mask
            if mots_vis:
                gt_file_mots = os.path.join(
                    root_split_mots_path,
                    seq.replace('MOT17', 'MOTS20'),
                    'gt',
                    'gt.txt')
                if os.path.isfile(gt_file_mots):
                    mask_objects_per_frame = load_mots_gt(gt_file_mots)

                    for frame_id, frame_annotations in seq_annotations_per_frame.items():
                        mask_objects = mask_objects_per_frame[frame_id]
                        mask_object_bboxes = [rletools.toBbox(obj.mask) for obj in mask_objects]
                        mask_object_bboxes = torch.tensor(mask_object_bboxes).float()

                        frame_boxes = [a['bbox'] for a in frame_annotations]
                        frame_boxes = torch.tensor(frame_boxes).float()

                        # x,y,w,h --> x,y,x,y
                        frame_boxes[:, 2:] += frame_boxes[:, :2]
                        mask_object_bboxes[:, 2:] += mask_object_bboxes[:, :2]

                        mask_iou = box_iou(mask_object_bboxes, frame_boxes)

                        mask_indices, frame_indices = linear_sum_assignment(-mask_iou)
                        for m_i, f_i in zip(mask_indices, frame_indices):
                            if mask_iou[m_i, f_i] < 0.5:
                                continue

                            if not frame_annotations[f_i]['visibility']:
                                frame_annotations[f_i]['ignore'] = 0

    # max objs per image
    num_objs_per_image = {}
    for anno in annotations['annotations']:
        image_id = anno["image_id"]

        if image_id in num_objs_per_image:
            num_objs_per_image[image_id] += 1
        else:
            num_objs_per_image[image_id] = 1

    print(f'max objs per image: {max(list(num_objs_per_image.values()))}')

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


def check_coco_from_mot(split='train'):
    """
    Visualize generated COCO data. Only used for debugging.
    """
    coco_dir = os.path.join(DATA_ROOT, split)
    annotation_file = os.path.join(coco_dir, 'annotations.json')

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(catNms=['pedestrian'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    index = np.random.randint(0, len(img_ids))
    img = coco.loadImgs(img_ids[index])[0]

    i = io.imread(os.path.join(coco_dir, img['file_name']))

    plt.imshow(i)
    plt.axis('off')
    ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    coco.showAnns(anns, draw_bbox=True)
    plt.savefig('annotations.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO from MOT.')
    parser.add_argument('--mots20', action='store_true')
    args = parser.parse_args()

    ant_seqs_names = list(ANT_SEQS_INFO.keys())

    generate_coco_from_mot(
        'train',
        seqs_names= ["OU10B3L2In_0", "OU10B1L1Out_0"],
        root_split='train_split'
    )

    generate_coco_from_mot(
        'val',
        seqs_names=["OU10B3L2Out_0", "OU10B3L3Out_0"],
        root_split='val_split'
    )

    generate_coco_from_mot(
        'test',
        seqs_names=["OU10B3L1In_0", "OU10B2L1In_0"],
        root_split='test_split'
    )