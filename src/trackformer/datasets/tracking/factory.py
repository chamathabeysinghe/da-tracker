# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union

from torch.utils.data import ConcatDataset

from .mot_wrapper import MOT17Wrapper, MOTS20Wrapper, AntWrapper
from .demo_sequence import DemoSequence

DATASETS = {}

# Fill all available datasets, change here to modify / add new datasets.
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))

# for split in ["OU10B3L2In_0", "OU10B1L1Out_0", "OU10B3L3In_0", "OU10B2L2In_0", "OU10B2L3Out_0", "OU10B2L2Out_0",
#               "OU10B2L3In_0", "OU10B1L2In_0", "OU10B1L3Out_0", "OU10B1L2Out_0", "OU10B1L3In_0", "OU50B1L1In_0",
#               "OU50B1L1Out_0", "OU50B1L2In_0", "OU50B2L2Out_0", "OU50B1L3In_0", "OU50B1L3Out_0", "OU50B1L2Out_0",
#               "OU50B2L2In_0", "OU50B2L3In_0",
#               "OU10B3L2Out_0", "OU10B3L3Out_0", "OU10B1L1In_0", "OU50B3L2In_0", "OU50B3L2Out_0", "OU50B3L3Out_0"]:
for split in ["CU10L1B1In_0", "CU10L1B1Out_0", "CU25L1B1Out_0", "CU25L1B1In_0", "CU10L1B4In_0", "CU10L1B4Out_0",
              "CU25L1B4In_0", "CU25L1B4Out_0", "CU10L1B5In_0", "CU10L1B6Out_0", "CU50L1B6In_0", "CU50L1B6Out_0",
              "CU15L1B1Out_0", "CU20L1B1In_0", "CU15L1B4Out_0", "CU20L1B4In_0", "CU10L1B5Out_0", "CU30L1B6In_0",
              "CU15L1B1In_0", "CU20L1B1Out_0", "CU15L1B4In_0", "CU20L1B4Out_0", "CU30L1B6Out_0", "CU10L1B6In_0",
              "OU10B1L1Out_0", "OU50B1L1In_0", "OU10B1L2Out_0", "OU50B1L2Out_0", "OU10B1L3In_0", "OU50B1L3In_0",
              "OU10B2L1Out_0", "OU50B2L1In_0", "OU10B2L2Out_0", "OU50B2L2In_0", "OU10B2L3In_0", "OU10B2L3Out_0",
              "OU50B2L3In_0", "OU50B3L1In_0", "OU10B3L2In_0", "OU10B3L2Out_0", "OU50B3L2In_0", "OU10B3L3In_0",
              "OU10B3L3Out_0", "OU50B3L3Out_0", "OU10B1L1In_0", "OU50B1L2In_0", "OU50B1L3Out_0", "OU10B2L2In_0",
              "OU10B3L1Out_0", "OU50B3L3In_0", "OU50B1L1Out_0", "OU10B1L2In_0", "OU10B1L3Out_0", "OU10B2L1In_0",
              "OU50B2L2Out_0", "OU10B3L1In_0", "OU50B3L2Out_0"]:
    name = split
    DATASETS[name] = (
        lambda kwargs, split=split: AntWrapper(split, **kwargs))


DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])


class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.

        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
