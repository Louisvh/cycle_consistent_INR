import argparse
from datetime import datetime
from models import models
import SimpleITK as sitk
import torch
import numpy as np
from tqdm import tqdm
from utils import general

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='New image pair optimization')

    parser.add_argument('--img_source', required=True,
                        help='Path to the source image file (i.e. image source, coordinate target)')
    parser.add_argument('--mask_source', required=False, default=None, help='Path to the source mask file')
    parser.add_argument('--img_target', required=True,
                        help='Path to the target image file (i.e. coordinate source, image target)')
    parser.add_argument('--landmarks', required=False, default=None,
                        help='Landmarks in the target image, as a .csv file, zyx format')
    parser.add_argument('--mask_target', required=False, default=None, help='Path to the target mask file')
    parser.add_argument('--out_dir', required=False, default='./results/', help='directory to save results')
    parser.add_argument('--disable_loss_schedule', action='store_true', default=False,
                        help='disable cosine loss schedule')
    parser.add_argument('--iterations', default=1250, help='number of optimization iterations')
    parser.add_argument('--batch_size', default=10000, help='batch size during optimization')
    parser.add_argument('--inf_batch_size', type=int, default=20000, help='batch size during inference')
    parser.add_argument('--plotdim', type=int, default=0, help='dimension to slice for plotting')
    parser.add_argument('--plotloc', type=float, default=0., help='slice location for plotting [-1, 1]')
    parser.add_argument('--cycle_inf_order', type=int, default=2, help='order for the Taylor expansion')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use')
    parser.add_argument('--save_full_images', action='store_true', default=False,
                        help='compute and save the complete image and vector fields')

    return parser.parse_args()


def main():
    args = parse_args()

    img_source = sitk.ReadImage(args.img_source)
    imarr_source = torch.Tensor(sitk.GetArrayFromImage(img_source))
    img_target = sitk.ReadImage(args.img_target)
    imarr_target = torch.Tensor(sitk.GetArrayFromImage(img_target))
    if args.mask_source is not None:
        mask_source = sitk.ReadImage(args.mask_source)
        maskarr_source = sitk.GetArrayFromImage(mask_source)
    else:
        maskarr_source = np.ones_like(imarr_source.numpy())
    og_landmarks_voxspace = None
    if args.landmarks is not None:
        og_landmarks_voxspace = np.loadtxt(args.landmarks, delimiter=',')
    if args.mask_target is not None:
        mask_target = sitk.ReadImage(args.mask_target)
        maskarr_target = sitk.GetArrayFromImage(mask_target)
    else:
        maskarr_target = np.ones_like(imarr_target.numpy())

    kwargs = {
        "verbose": True,
        "hyper_regularization": False,
        "raw_jacobian_regularization": False,
        "jacobian_regularization": True,
        "bending_regularization": False,
        "network_type": "SIREN",
        "save_folder": args.out_dir,
        "mask": maskarr_target,
        "mask_2": maskarr_source,
        "cycle_alpha": 1e-3,
        "optimizer": 'adam',
        "lr": 0.0001,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "cycle_loss_schedule": not args.disable_loss_schedule,
        "epochs": int(args.iterations),
        "cycle_loss_delay": 0,
        "layers": [3, 256, 256, 256, 3],
    }

    if kwargs["epochs"] < 2500 or kwargs["cycle_loss_schedule"]:
        print(f'Using short schedule for demo purposes.')
        print(f'To reproduce the results in the paper, use 2500 epochs with static lr.')

    ImpReg = models.ImplicitRegistrator(imarr_source, imarr_target, **kwargs)
    ImpReg.fit()

    filename = f'{args.img_source.split("/")[-1]}_{args.img_target.split("/")[-1]}.pth'
    ImpReg.savenets(filename, args.out_dir)

    plotdim = args.plotdim
    sliceshape = [imarr_source.shape[i] for i in range(3) if i != plotdim]
    field_slices, normed_im_slices = ImpReg.generate_valslice(plotdim, sliceshape, loc=args.plotloc, inf_order=args.cycle_inf_order)
    inplane_field_fw = np.delete(field_slices[0], plotdim, axis=-1)
    inplane_field_bw = np.delete(field_slices[1], plotdim, axis=-1)
    uncertnorm_fw = np.linalg.norm(field_slices[2], axis=-1)
    uncertnorm_bw = np.linalg.norm(field_slices[3], axis=-1)
    inplane_field_slices = (inplane_field_fw, inplane_field_bw, uncertnorm_fw, uncertnorm_bw)

    inplane_og_landmarks, inplane_prop_landmarks = None, None
    if og_landmarks_voxspace is not None:
        _, prop_landmarks, _ = general.compute_landmarks_cycle(ImpReg.network, ImpReg.network_rev,
                                                                       og_landmarks_voxspace, imarr_source.shape)
        np.savetxt(f'{args.out_dir}/propagated_{args.landmarks.split("/")[-1]}', prop_landmarks, delimiter=',', fmt='%.3f')
        inplane_og_landmarks = np.delete(og_landmarks_voxspace, plotdim, axis=-1)
        inplane_prop_landmarks = np.delete(prop_landmarks, plotdim, axis=-1)
    plot_slices(inplane_field_slices, normed_im_slices, inplane_og_landmarks, inplane_prop_landmarks)

    # Save the complete image and vector fields
    if args.save_full_images:
        print(f'Generating and saving full vector fields, images and uncertainties (slice-by-slice, may take a while)')
        full_field_fw, full_field_bw = [], []
        full_uncert_fw, full_uncert_bw = [], []
        full_pred_fw, full_pred_bw = [], []
        for slicenum in tqdm(range(imarr_source.shape[plotdim])):
            sliceloc = (slicenum / (imarr_source.shape[plotdim]-1)) * 2 - 1
            field_slices, im_slices = ImpReg.generate_valslice(plotdim, sliceshape, sliceloc, max_batch=args.inf_batch_size, normalize=False, inf_order=args.cycle_inf_order)
            full_field_fw.append(field_slices[0])
            full_field_bw.append(field_slices[1])
            full_pred_fw.append(im_slices[2])
            full_pred_bw.append(im_slices[3])
            full_uncert_fw.append(field_slices[2])
            full_uncert_bw.append(field_slices[3])
        full_field_fw = np.stack(full_field_fw, axis=plotdim)
        full_field_bw = np.stack(full_field_bw, axis=plotdim)
        voxspace_field_fw = full_field_fw * np.array([(0.5 * (s-1)) for s in imarr_source.shape])
        voxspace_field_bw = full_field_bw * np.array([(0.5 * (s-1)) for s in imarr_source.shape])
        full_pred_fw = np.stack(full_pred_fw, axis=plotdim)
        full_pred_bw = np.stack(full_pred_bw, axis=plotdim)
        full_uncert_fw = np.stack(full_uncert_fw, axis=plotdim)
        full_uncert_bw = np.stack(full_uncert_bw, axis=plotdim)
        voxspace_uncert_fw = full_uncert_fw * np.array([(0.5 * (s-1)) for s in imarr_source.shape])
        voxspace_uncert_bw = full_uncert_bw * np.array([(0.5 * (s-1)) for s in imarr_source.shape])
        # save both fields
        sitk.WriteImage(sitk.GetImageFromArray(voxspace_field_fw), f'{args.out_dir}/field_fw_{filename.replace(".pth", ".nii.gz")}')
        sitk.WriteImage(sitk.GetImageFromArray(voxspace_field_bw), f'{args.out_dir}/field_bw_{filename.replace(".pth", ".nii.gz")}')
        # save both images
        sitk.WriteImage(sitk.GetImageFromArray(full_pred_fw), f'{args.out_dir}/pred_fw_{filename.replace(".pth", ".nii.gz")}')
        sitk.WriteImage(sitk.GetImageFromArray(full_pred_bw), f'{args.out_dir}/pred_bw_{filename.replace(".pth", ".nii.gz")}')
        # save both uncertainty maps
        sitk.WriteImage(sitk.GetImageFromArray(voxspace_uncert_fw), f'{args.out_dir}/uncert_fw_{filename.replace(".pth", ".nii.gz")}')
        sitk.WriteImage(sitk.GetImageFromArray(voxspace_uncert_bw), f'{args.out_dir}/uncert_bw_{filename.replace(".pth", ".nii.gz")}')


def plot_slices(field_slices, normed_im_slices, og_landmarks, prop_landmarks):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    voxfield_fw = field_slices[0] * np.array([(0.5 * (s-1)) for s in normed_im_slices[0].shape])
    jointmask = 1.0 * ((normed_im_slices[-2]+normed_im_slices[-1]) > 0)
    plt.imshow((normed_im_slices[0] - normed_im_slices[1]) * jointmask, cmap='bwr', vmin=-1, vmax=1)
    for i in range(0, normed_im_slices[0].shape[0], 10):
        for j in range(0, normed_im_slices[0].shape[1], 10):
            plt.plot([j, j + voxfield_fw[i, j, 1]], [i, i + voxfield_fw[i, j, 0]], c='k', linewidth=0.5)
    if og_landmarks is not None:
        for lmi in range(og_landmarks.shape[0]):
            plt.plot([og_landmarks[lmi, 1], prop_landmarks[lmi, 1]], [og_landmarks[lmi, 0], prop_landmarks[lmi, 0]],
                     c='k', linewidth=1, zorder=1)
            plt.scatter(og_landmarks[lmi, 1], og_landmarks[lmi, 0], c='b', s=5, zorder=2)
            plt.scatter(prop_landmarks[lmi, 1], prop_landmarks[lmi, 0], c='xkcd:red', s=5, marker='x', zorder=3)
    plt.title('Initial diff image with deformation vectors')
    plt.ylim(plt.ylim()[::-1])

    plt.subplot(2, 3, 2)
    plt.imshow((normed_im_slices[0]-normed_im_slices[2]) * normed_im_slices[-2], cmap='bwr', vmin=-1, vmax=1)

    plt.title('Result diff image ')
    plt.ylim(plt.ylim()[::-1])

    plt.subplot(2, 3, 3)
    clipval = .1
    plt.imshow(field_slices[2], cmap='bwr', vmin=-clipval, vmax=clipval)
    plt.title('Uncertainty')
    plt.ylim(plt.ylim()[::-1])

    # Plot the normed image slices
    plt.subplot(2, 3, 4)
    plt.imshow(normed_im_slices[1], cmap='gray')
    plt.title('Source Image Slice')
    if og_landmarks is not None:
        for lmi in range(og_landmarks.shape[0]):
            plt.scatter(prop_landmarks[lmi, 1], prop_landmarks[lmi, 0], c='r', s=5, marker='x')
    plt.ylim(plt.ylim()[::-1])

    plt.subplot(2, 3, 5)
    plt.imshow(normed_im_slices[2], cmap='gray')
    plt.title('Transformed Source Image Slice')
    plt.ylim(plt.ylim()[::-1])

    plt.subplot(2, 3, 6)
    plt.imshow(normed_im_slices[0], cmap='gray')
    plt.title('Target Image Slice')
    if og_landmarks is not None:
        for lmi in range(og_landmarks.shape[0]):
            plt.scatter(og_landmarks[lmi, 1], og_landmarks[lmi, 0], c='b', s=5)
    plt.ylim(plt.ylim()[::-1])

    plt.show()

if __name__ == "__main__":
    main()
