# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         utilities.py
# Description:  Contains a variety of useful functions.
# ===========================================================================
import torch
import warnings
from bisect import bisect_right
import math


class Utilities:
    @staticmethod
    @torch.no_grad()
    def get_layer_norms(model):
        """Computes L1, L2, Linf norms of all layers of model"""
        layer_norms = dict()
        for name, param in model.named_parameters():
            idxLastDot = name.rfind(".")
            layer_name, weight_type = name[:idxLastDot], name[idxLastDot + 1:]
            if layer_name not in layer_norms:
                layer_norms[layer_name] = dict()
            layer_norms[layer_name][weight_type] = dict(
                L1=float(torch.norm(param, p=1)),
                L2=float(torch.norm(param, p=2)),
                Linf=float(torch.norm(param, p=float('inf')))
            )

        return layer_norms

    @staticmethod
    @torch.no_grad()
    def get_model_norm_square(model):
        """Get L2 norm squared of parameter vector. This works for a pruned model as well."""
        squared_norm = 0.
        param_list = ['weight', 'bias']
        for name, module in model.named_modules():
            for param_type in param_list:
                if hasattr(module, param_type) and not isinstance(getattr(module, param_type), type(None)):
                    param = getattr(module, param_type)
                    squared_norm += torch.norm(param, p=2) ** 2
        return float(squared_norm)

    @staticmethod
    @torch.no_grad()
    def get_number_of_parameters(moduleList):
        """Get number of parameters of params in moduleList."""
        n = 0
        for module, param_type in moduleList:
            param = getattr(module, param_type)
            n += int(param.numel())
        return n

    @staticmethod
    @torch.no_grad()
    def get_number_of_filters(moduleList):
        """Get number of conv filters of model. This only work when moduleList contains valid (nn.Conv2d, 'weight')
        pairs """
        return sum([getattr(*(module, param_type)).shape[0] for module, param_type in moduleList])

    @staticmethod
    @torch.no_grad()
    def replace_layers_by_decomposition(model, svdSparsity, layer_type):
        """Replaces all linear or convolutional modules by their truncated SVD"""
        assert layer_type in ['linear', 'conv']
        if layer_type == 'linear':
            originalLayerClass = torch.nn.Linear
            decomposedLayerClass = DecomposedLinear
        elif layer_type == 'conv':
            originalLayerClass = torch.nn.Conv2d
            decomposedLayerClass = DecomposedConv2d

        for motherName, mother in list(model.named_modules()):
            for childName, child in list(mother.named_children()):
                if isinstance(child, originalLayerClass) and not isinstance(child, decomposedLayerClass):
                    child_device = getattr(child, 'weight').device
                    new_child = decomposedLayerClass(originalLayer=child, svdSparsity=svdSparsity).to(child_device)
                    setattr(mother, childName, new_child)

    @staticmethod
    @torch.jit.script
    @torch.no_grad()
    def SVD_eigval(W):
        """Computes the first Singular value by LOBPCG of W.T*W"""
        assert len(W.shape) == 2, "W must be a matrix"

        spdMatrix = torch.mm(W.t(), W)
        eVal, eVec = torch.lobpcg(A=spdMatrix, k=1)
        sigma = torch.sqrt(eVal)  # singular value
        v = eVec  # right singular vector
        u = 1. / sigma * torch.mm(W, v)
        return u.squeeze(), v.squeeze(), sigma

    @staticmethod
    @torch.jit.script
    @torch.no_grad()
    def SVD_power_iteration(W):
        """Requires the gap assumption, i.e. the first Eval is separated from the others by a sufficiently large gap.
        Implementation mostly taken from Pytorch Spectral, explanation why this works can be found here:
        Simple Algorithms for the Partial Singular Value Decomposition (J.C. Nash)"""
        assert len(W.shape) == 2, "W must be a matrix"
        n_power_iterations = 20
        n, m = W.shape

        # Initialize random vectors u and v
        u = torch.nn.functional.normalize(W.new_empty(n).normal_(0., 1.), dim=0)
        v = torch.nn.functional.normalize(W.new_empty(m).normal_(0., 1.), dim=0)

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            v = torch.nn.functional.normalize(torch.mv(W.t(), u), dim=0, out=v)
            u = torch.nn.functional.normalize(torch.mv(W, v), dim=0, out=u)

        sigma = torch.dot(u, torch.mv(W, v))
        return u, v, sigma

    @staticmethod
    @torch.no_grad()
    def SVD_block_power_iteration(W, k=1):
        """Block Power Iteration using the QR decomposition"""
        assert len(W.shape) == 2, "W must be a matrix"
        assert k <= min(W.shape)
        n_power_iterations = 10
        n, m = W.shape

        # Initialize block vectors
        V = torch.nn.functional.normalize(W.new_empty(m, k).normal_(0., 1.), dim=1)
        for _ in range(n_power_iterations):
            Q, R = torch.linalg.qr(torch.mm(W, V))
            U = Q
            Q, R = torch.linalg.qr(torch.mm(W.t(), U))
            V = Q
        sigma = torch.diag(R)

        # It might be that the singular values have the wrong sign
        U = torch.sign(sigma) * U
        sigma = torch.abs(sigma)
        return U, V.t(), sigma

    @staticmethod
    @torch.no_grad()
    def SVD_partial(W, k=1):
        """Computes the first k singular value/vector pairs. Note: Right now this needs to compute the full SVD, hence
        yielding no speedup for small k."""
        assert len(W.shape) == 2, "W must be a matrix"
        assert k <= min(W.shape)

        # Compute full SVD
        U, S, V_t = torch.linalg.svd(W, full_matrices=False)
        n_singvals_to_keep = k  # Number of singular values to keep

        # Truncate matrices
        V_t = V_t[:n_singvals_to_keep, :]  # Keep only the first rows
        S = S[:n_singvals_to_keep]
        U = U[:, :n_singvals_to_keep]  # Keep only the first columns

        return U, V_t, S


class FixedLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Just uses the learning rate given by a list
    """

    def __init__(self, optimizer, lrList, last_epoch=-1):
        self.lrList = lrList

        super(FixedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.lrList[self.last_epoch] for group in self.optimizer.param_groups]


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()


class ChainedSchedulers(torch.optim.lr_scheduler.ChainedScheduler):
    """
    Repairs ChainedScheduler to avoid a known bug that makes it into the pytorch release soon
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(ChainedSchedulers, self).__init__(**kwargs)


class CyclicLRAdaptiveBase(torch.optim.lr_scheduler.CyclicLR):

    def __init__(self, base_lr_scale_fn=None, **kwargs):
        self.base_lr_scale_fn = base_lr_scale_fn
        super(CyclicLRAdaptiveBase, self).__init__(**kwargs)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        # Adjust the base lrs
        if self.base_lr_scale_fn:
            for entry_idx in range(len(self.base_lrs)):
                self.base_lrs[entry_idx] = self.max_lrs[entry_idx] * self.base_lr_scale_fn(cycle)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs


class DecomposedLinear(torch.nn.Module):
    """Converts a single Linear Layer into two Linear layers using Truncated SVD. For more information see
    https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning"""

    def __init__(self, originalLayer, svdSparsity):
        super().__init__()
        assert isinstance(originalLayer, torch.nn.Linear)
        assert 0 <= svdSparsity < 1
        self.compute_decomposition(originalLayer=originalLayer, svdSparsity=svdSparsity)

    @torch.no_grad()
    def compute_decomposition(self, originalLayer, svdSparsity):
        # Computes the SVD, truncates and creates two new linear layers
        W = originalLayer.weight
        b = originalLayer.bias  # None if not existing
        n, m = W.shape  # Maps from R^m to R^n

        # Compute SVD
        U, S, V_t = torch.linalg.svd(W, full_matrices=False)
        n_singvals_to_keep = max(1, int((1 - svdSparsity) * S.shape[0]))  # Number of singular values to keep

        # Truncate matrices
        V_t = V_t[:n_singvals_to_keep, :]  # Keep only the first rows
        S = S[:n_singvals_to_keep]
        U = U[:, :n_singvals_to_keep]  # Keep only the first columns

        # Initialize two new layers of shape
        self.first = torch.nn.Linear(in_features=m, out_features=n_singvals_to_keep, bias=False)
        self.second = torch.nn.Linear(in_features=n_singvals_to_keep, out_features=n, bias=(b is not None))

        # Set the parameters as computed in the SVD
        self.first.weight.copy_(torch.matmul(torch.diag(S), V_t))
        self.second.weight.copy_(U)
        if b is not None:
            self.second.bias.copy_(b)

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return out


class DecomposedConv2d(torch.nn.Module):
    """Converts a single Conv2d layer into two Conv2d layers by exploiting truncated SVD. The idea is
     to view the 4d-convolution as a matrix by flattening in_channels, and (k,k)."""

    def __init__(self, originalLayer, svdSparsity):
        super().__init__()
        assert isinstance(originalLayer, torch.nn.Conv2d)
        assert 0 <= svdSparsity < 1
        self.compute_decomposition(originalLayer=originalLayer, svdSparsity=svdSparsity)

    @torch.no_grad()
    def compute_decomposition(self, originalLayer, svdSparsity):
        # Save information of originalLayer
        W = originalLayer.weight
        b = originalLayer.bias  # None if not existing
        c_out, c_in, k1, k2 = W.shape
        stride, padding = originalLayer.stride, originalLayer.padding
        n, m = W.flatten(start_dim=1).shape  # Maps from R^m to R^n

        # Compute SVD
        U, S, V_t = torch.linalg.svd(W.flatten(start_dim=1), full_matrices=False)
        n_singvals_to_keep = max(1, int((1 - svdSparsity) * S.shape[0]))  # Number of singular values to keep

        # Truncate matrices
        V_t = V_t[:n_singvals_to_keep, :]  # Keep only the first rows
        S = S[:n_singvals_to_keep]
        U = U[:, :n_singvals_to_keep]  # Keep only the first columns

        # Initialize two new layers of shape
        self.first = torch.nn.Conv2d(in_channels=c_in, out_channels=n_singvals_to_keep,
                                     kernel_size=(k1, k2), stride=stride, padding=padding, bias=False)
        self.second = torch.nn.Conv2d(in_channels=n_singvals_to_keep, out_channels=c_out,
                                      kernel_size=(1, 1), bias=(b is not None))

        # Set the parameters as computed in the SVD
        self.first.weight.copy_(torch.matmul(torch.diag(S), V_t).view(self.first.weight.shape))
        self.second.weight.copy_(U.view(self.second.weight.shape))
        if b is not None:
            self.second.bias.copy_(b)

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return out
