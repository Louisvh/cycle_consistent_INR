import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import tqdm

from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""

        self.set_default_arguments()
        self.parse_kwargs(kwargs)

        self.initialize_networks()
        self.initialize_optimizer()

        self.moving_image = moving_image
        self.fixed_image = fixed_image
        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()

        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
            self.mask, self.fixed_image.shape
        )
        self.possible_coordinate_tensor_reverse = general.make_masked_coordinate_tensor(
            self.mask_2, self.fixed_image.shape
        )

    def initialize_networks(self):
        if self.network_type == "MLP":
            self.network = networks.MLP(self.layers)
            self.network_rev = networks.MLP(self.layers)
        elif self.network_type == "SIREN":
            self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            self.network_rev = networks.Siren(self.layers, self.weight_init, self.omega)
        else:
            print(f'network type {self.network_type} not found!')
            exit
        if self.verbose:
            print(
                "Network contains {} trainable parameters.".format(
                    general.count_parameters(self.network)
                )
            )
        if self.network_from_file is not None:
            self.network.load_state_dict(torch.load(self.network_from_file))
            self.network_rev.load_state_dict(torch.load(self.network_from_file.replace('F', 'B')))
        if self.gpu:
            self.network.cuda()
            self.network_rev.cuda()

    def initialize_optimizer(self):
        # Choose the optimizer
        m_params = list(self.network.parameters()) + list(self.network_rev.parameters())
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                m_params, lr=self.lr, momentum=self.momentum
            )
        elif self.optimizer_arg.lower() == "adamw":
            self.optimizer = optim.AdamW(m_params, lr=self.lr)

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(m_params, lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(m_params, lr=self.lr)
        else:
            self.optimizer = optim.SGD(
                m_params, lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )
        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

    def parse_kwargs(self, kwargs):
        # Check if all kwargs keys are valid (this checks for typos)
        for kwarg in kwargs:
            if kwarg not in self.args.keys():
                print(f'\n---\nwarning: kwarg {kwarg} not found in args!!\n---')
        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.onecycle_policy = kwargs["onecycle_policy"] if "onecycle_policy" in kwargs else self.args[
            "onecycle_policy"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )
        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )
        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        # Add slash to divide folder and filename
        self.save_folder += "/"
        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]
        self.cycle_loss_list = [0 for _ in range(self.epochs)]
        # Set seed
        self.seed = (
            kwargs["seed"] if "seed" in kwargs else self.args["seed"]
        )
        torch.manual_seed(self.seed)
        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )

        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]
        self.mask_2 = kwargs["mask_2"] if "mask_2" in kwargs else self.args["mask_2"]
        self.cycle_l1 = (
            kwargs["cycle_l1"]
            if "cycle_l1" in kwargs
            else self.args["cycle_l1"]
        )
        self.cycle_loss_schedule = (
            kwargs["cycle_loss_schedule"]
            if "cycle_loss_schedule" in kwargs
            else self.args["cycle_loss_schedule"]
        )
        self.cycle_loss_delay = (
            kwargs["cycle_loss_delay"]
            if "cycle_loss_delay" in kwargs
            else self.args["cycle_loss_delay"]
        )
        # Parse regularization kwargs
        self.raw_jacobian_regularization = (
            kwargs["raw_jacobian_regularization"]
            if "raw_jacobian_regularization" in kwargs
            else self.args["raw_jacobian_regularization"]
        )
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )
        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )
        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )
        # Set seed
        torch.manual_seed(self.seed)
        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )
        self.cycle_alpha = (
            kwargs["cycle_alpha"] if "cycle_alpha" in kwargs else self.args["cycle_alpha"]
        )
        self.cycle_in_mm = (
            kwargs["cycle_in_mm"]
            if "cycle_in_mm" in kwargs
            else self.args["cycle_in_mm"]
        )
        self.voxel_size = (
            kwargs["voxel_size"]
            if "voxel_size" in kwargs
            else self.args["voxel_size"]
        )
        self.voxel_size = torch.tensor(self.voxel_size).cuda()

    def cuda(self):
        """Move the model to the GPU."""
        self.network.cuda()
        self.network_rev.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["cycle_alpha"] = 1e-3
        self.args["cycle_in_mm"] = True
        self.args["voxel_size"] = (1,1,1)

        self.args["lr"] = 1e-4
        self.args["onecycle_policy"] = False
        self.args["batch_size"] = 10000
        self.args["layers"] = [3, 256, 256, 256, 3]

        self.args["raw_jacobian_regularization"] = False
        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["cycle_loss_delay"] = 0
        self.args["cycle_loss_schedule"] = False
        self.args["cycle_l1"] = False
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "SIREN"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        cycle_training = self.cycle_alpha > 0

        # Reset the gradient
        self.network.train()
        self.network_rev.train()

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
        )[: self.batch_size]
        indices_rev = torch.randperm(
            self.possible_coordinate_tensor_reverse.shape[0], device="cuda"
        )[: self.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        coordinate_tensor_rev = self.possible_coordinate_tensor_reverse[indices_rev, :]
        coordinate_tensor_rev = coordinate_tensor_rev.requires_grad_(True)
        fixed_coord_samples = general.fast_trilinear_interpolation(
            self.fixed_image,
            coordinate_tensor[:, 0],
            coordinate_tensor[:, 1],
            coordinate_tensor[:, 2],
        )
        moving_coord_samples = general.fast_trilinear_interpolation(
            self.moving_image,
            coordinate_tensor_rev[:, 0],
            coordinate_tensor_rev[:, 1],
            coordinate_tensor_rev[:, 2],
        )

        forward_estimate_rel = self.network(coordinate_tensor)
        forward_estimate = torch.add(forward_estimate_rel, coordinate_tensor)
        output = forward_estimate
        transformed_samples_fw = self.transform_no_add(forward_estimate)

        backward_estimate_rel = self.network_rev(coordinate_tensor_rev)
        backward_estimate = torch.add(backward_estimate_rel, coordinate_tensor_rev)
        output_rev = backward_estimate
        transformed_samples_bw = self.transform_no_add(backward_estimate, self.fixed_image)

        if cycle_training:
            cycled_output = self.network_rev(output)
            cycled_output_B = self.network(output_rev)

            # if -rev is a true inverse, these should cancel
            cycle_error_A = cycled_output + forward_estimate_rel
            cycle_error_B = cycled_output_B + backward_estimate_rel
            if self.cycle_in_mm:
                cycle_error_A = general.scale_coorddiffs_to_mm(cycle_error_A, self.fixed_image.shape) * self.voxel_size
                cycle_error_B = general.scale_coorddiffs_to_mm(cycle_error_B, self.moving_image.shape) * self.voxel_size

        # Compute the loss
        loss += self.criterion(transformed_samples_fw, fixed_coord_samples)
        loss += self.criterion(transformed_samples_bw, moving_coord_samples)
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        if cycle_training:
            if self.cycle_l1:
                cycle_loss = torch.mean(torch.linalg.vector_norm(cycle_error_A, axis=1)) + torch.mean(torch.linalg.vector_norm(cycle_error_B, axis=1))
            else:
                cycle_loss = torch.mean(torch.square(cycle_error_A)) + torch.mean(torch.square(cycle_error_B))
            if epoch > self.cycle_loss_delay:
                if self.cycle_loss_schedule:
                    cycle_alpha = (epoch-self.cycle_loss_delay)/(self.epochs-self.cycle_loss_delay) * self.cycle_alpha
                else:
                    cycle_alpha = self.cycle_alpha
                loss += cycle_alpha * cycle_loss

            if self.verbose:
                self.cycle_loss_list[epoch] = cycle_loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = forward_estimate_rel
        output_rel_rev = backward_estimate_rel

        # Regularization
        if self.raw_jacobian_regularization and self.alpha_jacobian > 0:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordinate_tensor_rev, output_rel_rev, batch_size=self.batch_size
            )
        if self.jacobian_regularization and self.alpha_jacobian > 0:
            loss += self.alpha_jacobian * regularizers.compute_balanced_jacobian_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss += self.alpha_jacobian * regularizers.compute_balanced_jacobian_loss(
                coordinate_tensor_rev, output_rel_rev, batch_size=self.batch_size
            )
        if self.hyper_regularization and self.alpha_hyper > 0:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordinate_tensor_rev, output_rel_rev, batch_size=self.batch_size
            )
        if self.bending_regularization and self.alpha_bending > 0:
            loss += self.alpha_bending * regularizers.compute_bending_energy(
                coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss += self.alpha_bending * regularizers.compute_bending_energy(
                coordinate_tensor_rev, output_rel_rev, batch_size=self.batch_size
            )

        # Perform the backpropagation and update the parameters accordingly

        self.optimizer.zero_grad()
        loss.backward()

        # update optimizer lr
        if self.onecycle_policy:
            if epoch < self.epochs/4:
                cur_lr = self.lr/1000 + (4*epoch/self.epochs*self.lr)
            elif epoch < self.epochs*0.8:
                cur_lr = self.lr - (1/0.8)*4/3*(epoch-self.epochs/4)/self.epochs * self.lr
            else:
                cur_lr = self.lr/200
            for g in self.optimizer.param_groups:
                g['lr'] = cur_lr

        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.seed)

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]
            self.cycle_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            self.training_iteration(i)

            if i % (max(epochs//50,1)) == 0 and self.verbose:
                self.save_losslogs()

        print('loss (start, middle, end)')
        print(self.loss_list[0], self.loss_list[epochs//2], self.loss_list[-1])

    def generate_valslice(self, plotdim, outshape, loc=0.1, only_fields=False, inf_order=1, max_batch=10000, normalize=True):
        xcoords = torch.arange(-1, 1 + 1e-6, 2 / (outshape[0] - 1))
        ycoords = torch.arange(-1, 1 + 1e-6, 2 / (outshape[1] - 1))
        coordgrid_x, coordgrid_y = torch.meshgrid(xcoords, ycoords)
        if plotdim == 0:
            coordgrid_3d = (torch.zeros_like(coordgrid_x) + loc, coordgrid_x, coordgrid_y)
        elif plotdim == 1:
            coordgrid_3d = (coordgrid_x, torch.zeros_like(coordgrid_x) + loc, coordgrid_y)
        else:
            coordgrid_3d = (coordgrid_x, coordgrid_y, torch.zeros_like(coordgrid_x) + loc)
        coordinates = torch.stack(coordgrid_3d, axis=-1).cuda()
        in_coords = coordinates.flatten(0, -2)
        in_coords = in_coords.requires_grad_(True)

        sampled_fixed_mask = torch.nn.functional.grid_sample(torch.Tensor(self.mask[None, None, :]).cuda(),
                                                             torch.flip(in_coords[None, None, None, :], [-1]),
                                                             align_corners=True).reshape(outshape)
        nonzero_mask_fw = (sampled_fixed_mask > 0).flatten()
        in_coords_fw = in_coords[nonzero_mask_fw]
        dvecresult = torch.zeros_like(in_coords)
        uncert_fw = torch.zeros_like(in_coords)

        sampled_moving_mask = torch.nn.functional.grid_sample(torch.Tensor(self.mask_2[None, None, :]).cuda(),
                                                                torch.flip(in_coords[None, None, None, :], [-1]),
                                                                align_corners=True).reshape(outshape)
        nonzero_mask_bw = (sampled_moving_mask > 0).flatten()
        in_coords_bw = in_coords[nonzero_mask_bw]
        dvecresult_rev = torch.zeros_like(in_coords)
        uncert_bw = torch.zeros_like(in_coords)

        if inf_order > 0:
            if in_coords_fw.shape[0] > 0:
                co_FS_fw, co_BFS_fw, transformed_offset_fw = general.compute_coordinates_cycle_autoretry(self.network, self.network_rev, in_coords_fw, max_batch=max_batch, inf_order=inf_order)
                coordresult_fw = co_FS_fw + 0.5 * transformed_offset_fw
                partial_dvecresult = coordresult_fw - in_coords_fw
                dvecresult[nonzero_mask_fw] = partial_dvecresult
                uncert_fw[nonzero_mask_fw] = transformed_offset_fw
            if in_coords_bw.shape[0] > 0:
                co_FS_bw, co_BFS_bw, transformed_offset_bw = general.compute_coordinates_cycle_autoretry(self.network_rev, self.network, in_coords_bw, max_batch=max_batch, inf_order=inf_order)
                coordresult_bw = co_FS_bw + 0.5 * transformed_offset_bw
                partial_dvecresult_rev = coordresult_bw - in_coords_bw
                dvecresult_rev[nonzero_mask_bw] = partial_dvecresult_rev
                uncert_bw[nonzero_mask_bw] = transformed_offset_bw
        else:
            if in_coords_fw.shape[0] > 0:
                partial_dvecresult = self.network(in_coords_fw)
                dvecresult[nonzero_mask_fw] = partial_dvecresult
            if in_coords_bw.shape[0] > 0:
                partial_dvecresult_rev = self.network_rev(in_coords_bw)
                dvecresult_rev[nonzero_mask_bw] = partial_dvecresult_rev

        fw_field = dvecresult.reshape(outshape + [3]).detach().cpu().numpy()
        bw_field = dvecresult_rev.reshape(outshape + [3]).detach().cpu().numpy()
        uncert_fw = uncert_fw.reshape(outshape + [3]).detach().cpu().numpy()
        uncert_bw = uncert_bw.reshape(outshape + [3]).detach().cpu().numpy()
        field_slices = (fw_field, bw_field, uncert_fw, uncert_bw)
        if only_fields:
            return field_slices

        pred_coords = in_coords + dvecresult
        pred_coords_rev = in_coords + dvecresult_rev

        sampled_fixed_im = torch.nn.functional.grid_sample(self.fixed_image[None, None, :],
                                                           torch.flip(in_coords[None, None, None, :], [-1]),
                                                           align_corners=True).reshape(outshape).detach().cpu().numpy()
        sampled_moving_im = torch.nn.functional.grid_sample(self.moving_image[None, None, :],
                                                            torch.flip(in_coords[None, None, None, :], [-1]),
                                                            align_corners=True).reshape(outshape).detach().cpu().numpy()
        sampled_pred_im = torch.nn.functional.grid_sample(self.moving_image[None, None, :],
                                                          torch.flip(pred_coords[None, None, None, :], [-1]),
                                                          align_corners=True).reshape(outshape)
        sampled_pred_im *= 1 * (sampled_fixed_mask > 0)
        sampled_pred_im = sampled_pred_im.detach().cpu().numpy()

        sampled_predrev_im = torch.nn.functional.grid_sample(self.fixed_image[None, None, :],
                                                             torch.flip(pred_coords_rev[None, None, None, :], [-1]),
                                                             align_corners=True).reshape(outshape)
        sampled_predrev_im *= 1 * (sampled_moving_mask > 0)
        sampled_predrev_im = sampled_predrev_im.detach().cpu().numpy()

        im_slices = (
            sampled_fixed_im, sampled_moving_im, sampled_pred_im, sampled_predrev_im,
            sampled_fixed_im - sampled_pred_im, sampled_moving_im - sampled_predrev_im,
            (sampled_fixed_mask.detach().cpu().numpy() > 0) * 1.0,
            (sampled_moving_mask.detach().cpu().numpy() > 0) * 1.0
        )
        if normalize:
            # ensure there is at least 1 zero-value in the masks to prevent issues with normalization
            im_slices[-1][0, 0] = 0
            im_slices[-2][0, 0] = 0

            shifted_im_slices = [x - x.min() for x in im_slices]
            normed_im_slices = [x / x.max() for x in shifted_im_slices]
        else:
            normed_im_slices = im_slices
        return field_slices, normed_im_slices

    def save_losslogs(self, experiment_name='default'):
        np.savetxt(f'{self.save_folder}/loss_log_{experiment_name}.txt',
                self.loss_list)
        np.savetxt(f'{self.save_folder}/data_loss_log_{experiment_name}.txt',
                self.data_loss_list)
        np.savetxt(f'{self.save_folder}/cycle_loss_log_{experiment_name}.txt',
                self.cycle_loss_list)

    def savenets(self, fname='default.pth', savedir='saved_models/'):
        f_fname = f'{savedir}F_{fname}'
        b_fname = f'{savedir}B_{fname}'
        torch.save(self.network.state_dict(), f_fname)
        torch.save(self.network_rev.state_dict(), b_fname)

