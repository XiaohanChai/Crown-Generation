# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn

import datasets
import models


def get_dataset(flags):
  print('[*] Dataset name: %s' % flags.name)
  if flags.name.lower() == 'shapenet' or flags.name.lower() == 'mask_crown' or flags.name.lower() == 'mask_crown_demo' or flags.name.lower() == 'mask_crown_750':
    return datasets.dualoctree_snet.get_shapenet_dataset(flags)
  else:
    raise ValueError
