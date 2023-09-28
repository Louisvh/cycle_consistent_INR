import numpy as np
import os
import torch
import SimpleITK as sitk


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 4)
    stds = np.round(stds, 4)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds


def compute_landmarks(network, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta

def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad

def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3, device='cuda')
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix

def compute_landmarks_cycle(network, network_rev, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordmarks_S = landmarks_to_coordmarks(landmarks_pre, image_size).cuda()
    
    co_FS_delta = network(coordmarks_S)
    co_FS = coordmarks_S + co_FS_delta

    co_BFS_delta = network_rev(co_FS)
    co_BFS = co_FS + co_BFS_delta

    dist_S_BFS = coordmarks_S - co_BFS
    jac_B_at_co_FS = compute_jacobian_matrix(co_FS, co_BFS, add_identity=False)

    offset_1st_der = torch.zeros_like(dist_S_BFS)
    offset_2nd_der = torch.zeros_like(dist_S_BFS)
    transformed_offset = torch.zeros_like(dist_S_BFS)
    transformed_offset_1st = torch.zeros_like(dist_S_BFS)
    for i in range(dist_S_BFS.shape[0]):
        b_i = dist_S_BFS[i,:]
        a_i = torch.inverse(jac_B_at_co_FS[i,:,:]).T
        offset_1st_der[i,:] = torch.matmul(b_i,a_i)
        transformed_offset_1st[i,:] = offset_1st_der[i,:]

    hessian_B_at_co_FS = compute_jacobian_matrix(co_FS, co_FS+offset_1st_der, add_identity=False)
    for i in range(dist_S_BFS.shape[0]):
        b_i = dist_S_BFS[i,:]
        a_i_inv = torch.inverse(hessian_B_at_co_FS[i,:,:])
        a_i = hessian_B_at_co_FS[i,:,:].T
        offset_2nd_der[i,:] = torch.matmul(b_i,a_i)

    transformed_offset = transformed_offset_1st + 0.5 * offset_2nd_der

    co_2invB_FS = co_FS+transformed_offset

    co_midpoint_FS_iBS = co_FS + 0.5*transformed_offset
    result_landmarks_FS = coordmarks_to_landmarks(co_FS, image_size).detach().cpu().numpy()
    result_landmarks_iBS = coordmarks_to_landmarks(co_2invB_FS, image_size).detach().cpu().numpy()
    result_landmarks_cyclecor = coordmarks_to_landmarks(co_midpoint_FS_iBS, image_size).detach().cpu().numpy()
    uncertainty = result_landmarks_FS - result_landmarks_iBS

    return result_landmarks_FS, result_landmarks_cyclecor, uncertainty

def landmarks_to_coordmarks(landmarks, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]
    coordmarks = torch.FloatTensor(landmarks / (scale_of_axes)) - 1.0
    return coordmarks

def coordmarks_to_landmarks(coordmarks, image_size):
    scale_of_axes = torch.FloatTensor([(0.5 * s) for s in image_size])
    if 'cuda' in str(coordmarks.device):
        scale_of_axes = scale_of_axes.cuda()
    landmarks = (coordmarks + 1.0) * (scale_of_axes)
    return landmarks

def scale_coorddiffs_to_mm(coorddiffs, image_size):
    scale_of_axes = torch.FloatTensor([(0.5 * s) for s in image_size])
    if 'cuda' in str(coorddiffs.device):
        scale_of_axes = scale_of_axes.cuda()
    coorddiffs_mm = (coorddiffs) * (scale_of_axes)
    return coorddiffs_mm

def load_image_DIRLab(variation=1, folder=r"D:\Data\DIRLAB\Case", separate_mask=False):
    # Size of data, per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + r"Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

    #with open(folder + r"Images/case" + str(variation) + "_T00.nii.gz", "rb") as f:
    #    data = np.fromfile(f, dtype)
    #image_insp = data.reshape(shape)
    fname_insp = folder + r"Images/case" + str(variation) + "_T00.nii.gz"
    image_insp = sitk.GetArrayFromImage(sitk.ReadImage(fname_insp))

    #with open(folder + r"Images/case" + str(variation) + "_T50.nii.gz", "rb") as f:
    #    data = np.fromfile(f, dtype)
    #image_exp = data.reshape(shape)
    fname_exp = folder + r"Images/case" + str(variation) + "_T50.nii.gz"
    image_exp = sitk.GetArrayFromImage(sitk.ReadImage(fname_exp))

    imgsitk_in = sitk.ReadImage(folder + r"Masks/case" + str(variation) + "_T00.nii.gz")
    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)
    if separate_mask:
        separatemask_0 = sitk.ReadImage(folder + r"Masks/auto_case" + str(variation) + "_T00.nii.gz")
        separatemask_50 = sitk.ReadImage(folder + r"Masks/auto_case" + str(variation) + "_T50.nii.gz")
        mask[0,:] = np.clip(sitk.GetArrayFromImage(separatemask_0), 0, 1)
        mask[5,:] = np.clip(sitk.GetArrayFromImage(separatemask_50), 0, 1)

    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + r"extremePhases/Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + r"extremePhases/Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing='ij')
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor.cuda()

    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    return coordinate_tensor
