from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

results = []
# file_names = [
#     "CU10L1B5Out_0",
# "CU15L1B1Out_0",
# "CU15L1B4Out_0",
# "CU20L1B1In_0",
# "CU20L1B4In_0",
# "CU30L1B6In_0"
# ]
file_names = [
    "CU10L1B6In_0",
    "CU15L1B1In_0",
    "CU15L1B4In_0",
    "CU20L1B1Out_0",
    "CU20L1B4Out_0",
    "CU30L1B6Out_0"
]
# file_names = [
#     "CU10L1B5Out_0",
#     "CU15L1B1Out_0",
#     "CU15L1B4Out_0",
#     "CU20L1B1In_0",
#     "CU20L1B4In_0",
#     "CU30L1B6In_0"
# ]
# file_names = [
#     "OU10B1L1In_0",
#     "OU10B1L3Out_0",
#     "OU10B2L2In_0",
#     "OU10B3L1Out_0",
#     "OU50B1L2In_0",
#     "OU50B2L2Out_0",
#     "OU50B3L3In_0",
#     "OU10B1L2In_0",
#     "OU10B2L1In_0",
#     "OU10B3L1In_0",
#     "OU50B1L1Out_0",
#     "OU50B1L3Out_0",
#     "OU50B3L2Out_0"
# ]
for f in file_names:
    annType = 'bbox'
    cocoGt = COCO(f'/Users/cabe0006/Projects/monash/trackformer/coco_evaluations/exp11d/raw_data/ground-truth-{f}.json')
    cocoDt = cocoGt.loadRes(
        f'/Users/cabe0006/Projects/monash/trackformer/coco_evaluations/exp11d/predicted_files/test-predictions-{f}.json')
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    results.append(f"{f} {cocoEval.stats[0]} {cocoEval.stats[1]} {cocoEval.stats[2]}")
    print(cocoEval.stats[:3])
    print('DONE')

for r in results:
    print(r)
