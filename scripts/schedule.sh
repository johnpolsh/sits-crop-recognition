#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python3 src/train_segmentation.py experiment=pastis_segmentation model=efswin_unet num_frames=6 data.batch_size=16 logger=tensorboard plugins=mixed_precision
python3 src/train_segmentation.py experiment=pastis_segmentation model=efswin_unet num_frames=8 data.batch_size=16 logger=tensorboard plugins=mixed_precision
python3 src/train_segmentation.py experiment=pastis_segmentation model=efswin_unet num_frames=12 data.batch_size=16 logger=tensorboard plugins=mixed_precision
python3 src/train_segmentation.py experiment=pastis_segmentation model=efswin_unet num_frames=16 data.batch_size=16 logger=tensorboard plugins=mixed_precision
