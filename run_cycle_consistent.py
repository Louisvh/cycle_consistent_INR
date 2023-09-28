from utils import general
from models import models
import os
import numpy as np
import sys

data_dir = os.path.expanduser("~/DATA/DIRLAB")
out_dir = "/tmp/"

if len(sys.argv) > 1:
    case_id = int(sys.argv[1])
else:
    case_id = 8

print(f'Optimizing DIR-Lab case id: {case_id}')

(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_full,
    voxel_size,
) = general.load_image_DIRLab(case_id, "{}/Case".format(data_dir))

kwargs = {}
kwargs["verbose"] = True
kwargs["hyper_regularization"] = False
kwargs["raw_jacobian_regularization"] = False
# Note: jacobian_regularization is the symmetric version. 
# For the version from the original IDIR, use "raw_jacobian_regulatization"
kwargs["jacobian_regularization"] = True
kwargs["bending_regularization"] = False
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = out_dir + str(case_id)
mask_exp = mask_full[0]
mask_insp = mask_full[5]
kwargs["mask"] = mask_exp
kwargs["mask_2"] = mask_insp
kwargs["cycle_alpha"] = 1e-3

kwargs["optimizer"] = 'adam'
kwargs["lr"] = 0.0001

from datetime import datetime
now = datetime.now()
kwargs["seed"] = now.microsecond

kwargs["batch_size"] = 10000
kwargs["cycle_loss_schedule"] = True #enables a simple cosine schedule. Set to False for static lr.
kwargs["epochs"] = 1250
kwargs["cycle_loss_delay"] = 0
kwargs["layers"] = [3, 256, 256, 256, 3]

if kwargs["epochs"] < 2500 or kwargs["cycle_loss_schedule"]:
    print(f'Using short schedule for demo purposes.')
    print(f'To reproduce the results in the paper, use 2500 epochs with static lr.')

ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
ImpReg.fit()

ImpReg.savenets(f'{case_id}_cc{kwargs["cycle_alpha"]}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth', out_dir)

print(f'\n----\nInspiration -> expiration results:')
print('\ncycle-trained, forward only:')
new_landmarks_orig, _ = general.compute_landmarks(
    ImpReg.network, landmarks_insp, image_size=img_insp.shape
)
accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
    new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
)
print(f"true error:       {accuracy_mean[0]:.3f}+-{accuracy_std[0]:.3f}mm")

print('\ncycle-trained, cycle-inference:')
new_landmarks_orig_fw, new_landmarks_orig_cycle, new_landmarks_orig_uncertainty = general.compute_landmarks_cycle(
    ImpReg.network, ImpReg.network_rev, landmarks_insp, image_size=img_insp.shape
)
cycle_accuracy_mean, cycle_accuracy_std = general.compute_landmark_accuracy(
    new_landmarks_orig_cycle, landmarks_exp, voxel_size=voxel_size
)
print(f'mean uncertainty: {np.linalg.norm(new_landmarks_orig_uncertainty*voxel_size, axis=-1).mean():.3f}mm')
print(f"true error:       {cycle_accuracy_mean[0]:.3f}+-{cycle_accuracy_std[0]:.3f}mm")

print(f'\n----\nExpiration -> inspiration results:')
print('\ncycle-trained, backward only:')
new_landmarks_orig_bw, _ = general.compute_landmarks(
    ImpReg.network_rev, landmarks_exp, image_size=img_insp.shape
)
accuracy_mean_bw, accuracy_std_bw = general.compute_landmark_accuracy(
    new_landmarks_orig_bw, landmarks_insp, voxel_size=voxel_size
)
print(f"true error:       {accuracy_mean_bw[0]:.3f}+-{accuracy_std_bw[0]:.3f}mm")

print('\ncycle-trained, cycle-inference backward:')
new_landmarks_orig_bw, new_landmarks_orig_cyclebw, new_landmarks_orig_uncertaintybw = general.compute_landmarks_cycle(
    ImpReg.network_rev, ImpReg.network, landmarks_exp, image_size=img_insp.shape
)
cycle_accuracy_mean_bw, cycle_accuracy_std_bw = general.compute_landmark_accuracy(
    new_landmarks_orig_cyclebw, landmarks_insp, voxel_size=voxel_size
)
print(f'mean uncertainty: {np.linalg.norm(new_landmarks_orig_uncertaintybw*voxel_size, axis=-1).mean():.3f}mm')
print(f"true error:       {cycle_accuracy_mean_bw[0]:.3f}+-{cycle_accuracy_std_bw[0]:.3f}mm")

