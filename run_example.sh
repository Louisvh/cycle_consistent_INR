#!/bin/sh

python cycle_inference_script.py \
    --img_source example_data/simulated_bowel_t0.nii.gz \
    --img_target example_data/simulated_bowel_t1.nii.gz \
    --landmarks example_data/simulated_bowel_landmarks.csv \
    --mask_source example_data/simulated_bowel_mask.nii.gz \
    --mask_target example_data/simulated_bowel_mask.nii.gz
