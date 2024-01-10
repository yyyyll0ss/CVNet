#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=2 multi_train.py --config-file config-files/WHU_VectorCD_hrnet48.yaml


