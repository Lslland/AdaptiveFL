# ===========================================================================
# Project:      Compression-aware Training of Neural Networks using Frank-Wolfe
# File:         constraints.py
# Description:  Constraints and useful functions
# ===========================================================================
import torch
import math
from sdfl_api.optimizers.utilities import Utilities as Utils

tolerance = 1e-10


# Auxiliary methods
@torch.no_grad()
def get_avg_init_norm(layer, param_type=None, ord=2, repetitions=100):
    """Computes the average norm of default layer initialization"""
    output = 0
    for _ in range(repetitions):
        layer.reset_parameters()
        output += torch.norm(getattr(layer, param_type), p=ord).item()
    return float(output) / repetitions


@torch.no_grad()
def get_model_init_norms(moduleList, global_constraint):
    if not global_constraint:
        # Return a value for each layer parameter
        init_norms = dict()
        for module, param_type in moduleList:
            if hasattr(module, 'reset_parameters'):
                param = getattr(module, param_type)
                shape = param.shape

                avg_norm = get_avg_init_norm(module, param_type=param_type, ord=2)
                if avg_norm == 0.0:
                    # Catch unlikely case that weight/bias is 0-initialized (e.g. BatchNorm does this)
                    avg_norm = 1.0
                init_norms[shape] = avg_norm
        return init_norms
    else:
        cum_avg_norm = 0.0
        for module, param_type in moduleList:
            if hasattr(module, 'reset_parameters'):
                avg_norm = get_avg_init_norm(module, param_type=param_type, ord=2)
                cum_avg_norm += avg_norm ** 2
        cum_avg_norm = math.sqrt(cum_avg_norm)
        return cum_avg_norm


@torch.no_grad()
def convert_lp_radius(r, N, in_ord=2, out_ord='inf'):
    """
    Convert between radius of Lp balls such that the ball of order out_order
    has the same L2 diameter as the ball with radius r of order in_order
    in N dimensions
    """
    # Convert 'inf' to float('inf') if necessary
    in_ord, out_ord = float(in_ord), float(out_ord)
    in_ord_rec = 0.5 if in_ord == 1 else 1.0 / in_ord
    out_ord_rec = 0.5 if out_ord == 1 else 1.0 / out_ord
    return r * N ** (out_ord_rec - in_ord_rec)


@torch.no_grad()
def get_lp_complementary_order(ord):
    """Get the complementary order"""
    ord = float(ord)
    if ord == float('inf'):
        return 1
    elif ord == 1:
        return float('inf')
    elif ord > 1:
        return 1.0 / (1.0 - 1.0 / ord)
    else:
        raise NotImplementedError(f"Order {ord} not supported.")


# Method to ensure initial feasibility of the parameters of a model
@torch.no_grad()
def make_feasible(constraintList):
    """Shift all model parameters inside the feasible region defined by constraints"""
    for constraint, param_list in constraintList:
        feasible = constraint.shift_inside(param_list)
        for p_idx, p in enumerate(param_list):
            p.copy_(feasible[p_idx])


# Methods for setting constraints
@torch.no_grad()
def set_lp_constraints(moduleList, global_constraint, p=2, value=300, mode='initialization'):
    """Create L_p constraints for each layer or globally,
     where p == ord, and value depends on mode (is radius, diameter, or
    factor to multiply average initialization norm with)"""
    constraintList = []

    # Compute average init norms if necessary
    if mode == 'initialization':
        # Either a dict or one value
        init_norms = get_model_init_norms(moduleList=moduleList, global_constraint=global_constraint)
    if not global_constraint:
        for module, param_type in moduleList:
            param = getattr(module, param_type)
            n = param.numel()
            if mode == 'radius':
                constraint = LpBall(n, p=p, l2_diameter=None, norm_radius=value)
            elif mode == 'diameter':
                constraint = LpBall(n, p=p, l2_diameter=value, norm_radius=None)
            elif mode == 'initialization':
                diameter = 2.0 * value * init_norms[param.shape]
                constraint = LpBall(n, p=p, l2_diameter=diameter, norm_radius=None)
            else:
                raise ValueError(f"Unknown mode {mode}")
            constraintList.append((constraint, [param]))
    else:
        n = Utils.get_number_of_parameters(moduleList)
        if mode == 'radius':
            constraint = LpBall(n, p=p, l2_diameter=None, norm_radius=value)
        elif mode == 'diameter':
            constraint = LpBall(n, p=p, l2_diameter=value, norm_radius=None)
        elif mode == 'initialization':
            diameter = 2.0 * value * init_norms
            constraint = LpBall(n, p=p, l2_diameter=diameter, norm_radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")
        constraintList.append((constraint, [getattr(module, param_type) for module, param_type in moduleList]))

    return constraintList


@torch.no_grad()
def set_k_constraints(moduleList, constr_type, global_constraint, k=1, value=300, mode='initialization',
                      adjust_diameter=False):
    """Create KSparsePolytope constraints for each layer or globally, and value depends on mode (is radius, diameter, or
    factor to multiply average initialization norm with). K can be given either as an absolute or relative value."""
    assert not (adjust_diameter and constr_type != 'k_sparse')
    if constr_type == 'k_sparse':
        baseConstraint = lambda n, k, l2_diameter, norm_radius: KSparsePolytope(n=n, k=k, l2_diameter=l2_diameter,
                                                                                norm_radius=norm_radius,
                                                                                adjust_diameter=adjust_diameter)
    elif constr_type == 'k_support':
        baseConstraint = KSupportNormBall
    elif constr_type == 'group_k_support':
        baseConstraint = GroupKSupportNormBall
    else:
        raise NotImplementedError

    constraintList = []

    # Compute average init norms if necessary
    if mode == 'initialization':
        # Either a dict or one value
        init_norms = get_model_init_norms(moduleList=moduleList, global_constraint=global_constraint)
    if not global_constraint:
        for module, param_type in moduleList:
            param = getattr(module, param_type)
            n = param.numel() if constr_type != 'group_k_support' else param.shape[0]
            layer_k = min(int(k * n), n)
            if k > 1:
                # An actual integer number was specified
                layer_k = min(int(k), n)
            layer_k = max(layer_k, 1)  # Must be at least one parameter per layer
            if mode == 'radius':
                constraint = baseConstraint(n, k=layer_k, l2_diameter=None, norm_radius=value)
            elif mode == 'diameter':
                constraint = baseConstraint(n, k=layer_k, l2_diameter=value, norm_radius=None)
            elif mode == 'initialization':
                diameter = 2.0 * value * init_norms[param.shape]
                constraint = baseConstraint(n, k=layer_k, l2_diameter=diameter, norm_radius=None)
            else:
                raise ValueError(f"Unknown mode {mode}")
            constraintList.append((constraint, [param]))
    else:
        n = Utils.get_number_of_parameters(moduleList) if constr_type != 'group_k_support' \
            else Utils.get_number_of_filters(moduleList)
        layer_k = min(int(k * n), n)
        if k > 1:
            # An actual integer number was specified
            layer_k = min(int(k), n)
        if mode == 'radius':
            constraint = baseConstraint(n, k=layer_k, l2_diameter=None, norm_radius=value)
        elif mode == 'diameter':
            constraint = baseConstraint(n, k=layer_k, l2_diameter=value, norm_radius=None)
        elif mode == 'initialization':
            diameter = 2.0 * value * init_norms
            constraint = baseConstraint(n, k=layer_k, l2_diameter=diameter, norm_radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")
        constraintList.append((constraint, [getattr(module, param_type) for module, param_type in moduleList]))

    return constraintList


@torch.no_grad()
def set_nuclear_constraints(moduleList, lmo_nuc_method, global_constraint, value=300, mode='initialization'):
    assert not global_constraint
    constraintList = []

    # Compute average init norms if necessary
    if mode == 'initialization':
        # Either a dict or one value
        init_norms = get_model_init_norms(moduleList=moduleList, global_constraint=global_constraint)
    if not global_constraint:
        for module, param_type in moduleList:
            param = getattr(module, param_type)
            n = param.numel()
            if mode == 'radius':
                constraint = NuclearNormBall(n, lmo_nuc_method, l2_diameter=None, norm_radius=value)
            elif mode == 'diameter':
                constraint = NuclearNormBall(n, lmo_nuc_method, l2_diameter=value, norm_radius=None)
            elif mode == 'initialization':
                diameter = 2.0 * value * init_norms[param.shape]
                constraint = NuclearNormBall(n, lmo_nuc_method, l2_diameter=diameter, norm_radius=None)
            else:
                raise ValueError(f"Unknown mode {mode}")
            constraintList.append((constraint, [param]))
    else:
        n = Utils.get_number_of_parameters(moduleList)
        if mode == 'radius':
            constraint = NuclearNormBall(n, lmo_nuc_method, l2_diameter=None, norm_radius=value)
        elif mode == 'diameter':
            constraint = NuclearNormBall(n, lmo_nuc_method, l2_diameter=value, norm_radius=None)
        elif mode == 'initialization':
            diameter = 2.0 * value * init_norms
            constraint = NuclearNormBall(n, lmo_nuc_method, l2_diameter=diameter, norm_radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")
        constraintList.append((constraint, [getattr(module, param_type) for module, param_type in moduleList]))

    return constraintList


@torch.no_grad()
def set_k_decomp_constraints(moduleList, constr_type, lmo_nuc_method, global_constraint, k, value=300,
                             mode='initialization'):
    assert not global_constraint
    if constr_type == 'k_sparse':
        baseConstraint = SpectralKSparsePolytope
    elif constr_type == 'k_support':
        baseConstraint = SpectralKSupportNormBall
    else:
        raise NotImplementedError

    constraintList = []

    # Compute average init norms if necessary
    if mode == 'initialization':
        # Either a dict or one value
        init_norms = get_model_init_norms(moduleList=moduleList, global_constraint=global_constraint)
    if not global_constraint:
        for module, param_type in moduleList:
            param = getattr(module, param_type)
            n, m = param.flatten(start_dim=1).shape
            n = min(n, m)  # maximal possible rank
            layer_k = min(int(k * n), n)
            if k > 1:
                # An actual integer number was specified
                layer_k = min(int(k), n)
            layer_k = max(layer_k, 1)  # Must be at least one parameter per layer
            if mode == 'radius':
                constraint = baseConstraint(n, layer_k, lmo_nuc_method, l2_diameter=None, norm_radius=value)
            elif mode == 'diameter':
                constraint = baseConstraint(n, layer_k, lmo_nuc_method, l2_diameter=value, norm_radius=None)
            elif mode == 'initialization':
                diameter = 2.0 * value * init_norms[param.shape]
                constraint = baseConstraint(n, layer_k, lmo_nuc_method, l2_diameter=diameter, norm_radius=None)
            else:
                raise ValueError(f"Unknown mode {mode}")
            constraintList.append((constraint, [param]))
    else:
        n = Utils.get_number_of_parameters(moduleList)
        layer_k = min(int(k * n), n)
        if k > 1:
            # An actual integer number was specified
            layer_k = min(int(k), n)
        layer_k = max(layer_k, 1)  # Must be at least one parameter per layer
        if mode == 'radius':
            constraint = baseConstraint(n, layer_k, lmo_nuc_method, l2_diameter=None, norm_radius=value)
        elif mode == 'diameter':
            constraint = baseConstraint(n, layer_k, lmo_nuc_method, l2_diameter=value, norm_radius=None)
        elif mode == 'initialization':
            diameter = 2.0 * value * init_norms
            constraint = baseConstraint(n, layer_k, lmo_nuc_method, l2_diameter=diameter, norm_radius=None)
        else:
            raise ValueError(f"Unknown mode {mode}")
        constraintList.append((constraint, [getattr(module, param_type) for module, param_type in moduleList]))

    return constraintList


@torch.no_grad()
def set_structured_k_constraints(moduleList, global_constraint, k=1, value=300, mode='initialization'):
    """This function just sets kSupportNormBall-constraints on the filters of Conv2d layers and L2Ball-constraints on
    all other parameters"""
    conv2dFilters = [(module, param_type) for module, param_type in moduleList if isinstance(module, torch.nn.Conv2d)
                     and param_type == 'weight']
    filterConstraintList = set_k_constraints(moduleList=conv2dFilters, constr_type='group_k_support',
                                             global_constraint=global_constraint, k=k, value=value, mode=mode)

    restParameters = [(module, param_type) for module, param_type in moduleList
                      if (module, param_type) not in conv2dFilters]
    restConstraintList = set_lp_constraints(moduleList=restParameters, global_constraint=global_constraint,
                                            p=2, value=value, mode=mode)

    return filterConstraintList + restConstraintList


@torch.no_grad()
def set_structured_decomp_constraints(moduleList, lmo_nuc_method, global_constraint, value=300, mode='initialization'):
    """This function just sets NuclearNormBall-constraints on the Conv2d layers and L2Ball-constraints on
    all other parameters"""
    assert not global_constraint, 'Cannot use global constraints for decomposition.'

    conv2dFilters = [(module, param_type) for module, param_type in moduleList if isinstance(module, torch.nn.Conv2d)
                     and param_type == 'weight']
    filterConstraintList = set_nuclear_constraints(moduleList=conv2dFilters, lmo_nuc_method=lmo_nuc_method,
                                                   global_constraint=global_constraint,
                                                   value=value, mode=mode)

    restParameters = [(module, param_type) for module, param_type in moduleList
                      if (module, param_type) not in conv2dFilters]
    restConstraintList = set_lp_constraints(moduleList=restParameters, global_constraint=global_constraint,
                                            p=2, value=value, mode=mode)

    return filterConstraintList + restConstraintList


@torch.no_grad()
def set_structured_k_decomp_constraints(moduleList, constr_type, lmo_nuc_method, global_constraint, k, value=300,
                                        mode='initialization'):
    """This function just sets k-Spectral-constraints on Conv2d layers and L2Ball-constraints on
    all other parameters"""
    assert not global_constraint, 'Cannot use global constraints for decomposition.'

    conv2dFilters = [(module, param_type) for module, param_type in moduleList if isinstance(module, torch.nn.Conv2d)
                     and param_type == 'weight']
    filterConstraintList = set_k_decomp_constraints(moduleList=conv2dFilters, constr_type=constr_type,
                                                    lmo_nuc_method=lmo_nuc_method,
                                                    global_constraint=global_constraint, k=k,
                                                    value=value, mode=mode)

    restParameters = [(module, param_type) for module, param_type in moduleList
                      if (module, param_type) not in conv2dFilters]
    restConstraintList = set_lp_constraints(moduleList=restParameters, global_constraint=global_constraint,
                                            p=2, value=value, mode=mode)

    return filterConstraintList + restConstraintList


# Constraint classes
class Constraint:
    """
    Parent/Base class for constraints.
    Important note: For pruning to work, Projections and LMOs must be such that 0 entries in the input receive 0 entries in the output.
    :param n: dimension of constraint parameter space
    """

    def __init__(self, n):
        self.n = n
        self._l2_diameter, self._norm_radius = None, None

    def get_diameter(self):
        return self._l2_diameter

    def get_radius(self):
        try:
            return self._norm_radius
        except:
            raise ValueError("Tried to get radius from a constraint without one")

    def lmo(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def shift_inside(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"

    def euclidean_project(self, x):
        assert x.numel() == self.n, f"shape {x.shape} does not match dimension {self.n}"


class LpBall(Constraint):
    """
    Constraint class for the n-dim Lp-Ball (p=ord) with L2-diameter diameter or radius.
    """

    def __init__(self, n, p=2, l2_diameter=None, norm_radius=None):
        super().__init__(n)
        self.p = float(p)
        self.q = get_lp_complementary_order(self.p)

        assert float(p) >= 1, f"Invalid order {p}"
        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given.")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2 * convert_lp_radius(norm_radius, self.n, in_ord=self.p, out_ord=2)
        elif norm_radius is None:
            self._norm_radius = convert_lp_radius(l2_diameter / 2.0, self.n, in_ord=2, out_ord=self.p)
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Computes and formats single_lmo solutions"""
        # Apply LMO
        v = self.single_lmo(torch.cat([g.flatten() for g in x]))
        v_list = []

        # Update parameters
        seen_elements = 0
        for p in x:
            n_p = p.numel()
            v_list.append(v[seen_elements:seen_elements + n_p].view(p.shape))
            seen_elements += n_p
        return v_list

    @torch.no_grad()
    def single_lmo(self, x):
        """Returns v with norm(v, self.p) <= r minimizing v*x"""
        if self.p == 1:
            v = torch.zeros_like(x)
            maxIdx = torch.argmax(torch.abs(x))
            v.flatten()[maxIdx] = -self._norm_radius * torch.sign(x.flatten()[maxIdx])
            return v
        elif self.p == 2:
            x_norm = float(torch.norm(x, p=2))
            if x_norm > tolerance:
                return -self._norm_radius * x.div(x_norm)
            else:
                return torch.zeros_like(x)
        elif self.p == float('inf'):
            return torch.full_like(x, fill_value=-self._norm_radius) * torch.sign(x)
        else:
            sgn_x = torch.sign(x).masked_fill_(x == 0, 1.0)
            absxqp = torch.pow(torch.abs(x), self.q / self.p)
            x_norm = float(torch.pow(torch.norm(x, p=self.q), self.q / self.p))
            if x_norm > tolerance:
                return -(self._norm_radius / x_norm) * sgn_x * absxqp
            else:
                return torch.zeros_like(x)

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the LpBall with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list
        x_norm = torch.norm(torch.cat([p.flatten() for p in x]), p=self.p)
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        else:
            return x

    @torch.no_grad()
    def euclidean_project(self, x):
        """Projects x to the closest (i.e. in L2-norm) point on the LpBall (p = 1, 2, inf) with radius r."""
        super().euclidean_project(x)
        if self.p == 1:
            x_norm = torch.norm(x, p=1)
            if x_norm > self._norm_radius:
                sorted = torch.sort(torch.abs(x.flatten()), descending=True).values
                running_mean = (torch.cumsum(sorted, 0) - self._norm_radius) / torch.arange(1, sorted.numel() + 1,
                                                                                            device=x.device)
                is_less_or_equal = sorted <= running_mean
                # This works b/c if one element is True, so are all later elements
                idx = is_less_or_equal.numel() - is_less_or_equal.sum() - 1
                return torch.sign(x) * torch.max(torch.abs(x) - running_mean[idx], torch.zeros_like(x))
            else:
                return x
        elif self.p == 2:
            x_norm = torch.norm(x, p=2)
            return self._norm_radius * x.div(x_norm) if x_norm > self._norm_radius else x
        elif self.p == float('inf'):
            return torch.clamp(x, min=-self._norm_radius, max=self._norm_radius)
        else:
            raise NotImplementedError(f"Projection not implemented for order {self.p}")


class KSupportNormBall(Constraint):
    """
    # Convex hull of all v s.t. ||v||_2 <= r, ||v||_0 <= k
    # This is a 'smooth' version of the KSparsePolytope, i.e. a mixture of KSparsePolytope and L2Ball allowing sparse activations of different magnitude
    # Note that the oracle will always return a vector v s.t. ||v||_0 == k, unless the input x satisfied ||x||_0 < k.
    # This Ball is due to Argyriou et al (2012)
    """

    def __init__(self, n, k=1, l2_diameter=None, norm_radius=None):
        super().__init__(n)
        self.k = min(k, n)
        # print(k, '==', n)
        # print("norm_radius", norm_radius, l2_diameter)
        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2.0 * norm_radius
        elif norm_radius is  None:
            self._norm_radius = l2_diameter / 2.0
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Computes and formats single_lmo solutions"""
        # Apply LMO
        # print(len(x))
        # print(x)
        v = self.single_lmo(torch.cat([g.flatten() for g in x]))
        # v = self.single_lmo_25(x[0])
        v_list = []

        seen_elements = 0
        for p in x:
            n_p = p.numel()
            v_list.append(v[seen_elements:seen_elements + n_p].view(p.shape))
            seen_elements += n_p
        return v_list

    @torch.no_grad()
    def single_lmo(self, x):
        """Returns v in KSupportNormBall w/ radius r minimizing v*x"""
        super().lmo(x)
        d = x.numel()

        # print("d: ", d)
        # print('k: ', self.k)
        # print(self.k, d)
        if self.k <= d // 2:
            # It's fast to get the maximal k values
            v = torch.zeros_like(x)
            # print(x)
            maxIndices = torch.topk(torch.abs(x.flatten()), k=self.k).indices
            v.flatten()[maxIndices] = x.flatten()[maxIndices]  # Projection to axis
            # v.flatten()[0:int(d * 0.25)] = x.flatten()[0:int(d * 0.25)]
        else:
            # Faster to get the n-d smallest values
            v = x.clone().detach()
            minIndices = torch.topk(torch.abs(x.flatten()), k=d - self.k, largest=False).indices
            v.flatten()[minIndices] = 0  # Projection to axis
        v_norm = float(torch.norm(v, p=2))
        # return v
        if v_norm > tolerance:
            # print(v_norm)
            # self._norm_radius = v_norm
            # print(-self._norm_radius * v.div(v_norm) )
            return -self._norm_radius * v.div(v_norm)  # Projection to Ball
        else:
            return torch.zeros_like(x)

    @torch.no_grad()
    def single_lmo_25(self, x):
        super().lmo(x)
        wk_size = list(x.size())
        # print(wk_size)
        v = torch.zeros_like(x)
        if len(wk_size) == 4:
            if wk_size[1] == 3:
                in_channel = wk_size[1]
            else:
                in_channel = int(wk_size[1] ) # 0.25
            out_channel = int(wk_size[0] ) # 0.25
            v[: out_channel, :in_channel, :, :] = x[:out_channel, :in_channel, :, :]
        elif len(wk_size) == 2:
            in_features = int(wk_size[1] ) # 0.25
            out_features = wk_size[0]
            v[:out_features, :in_features] = x[:out_features, :in_features]
        elif len(wk_size) == 1:
            num_features = int(wk_size[0] ) # 0.25
            v[:num_features] = x[:num_features]
        v = v.flatten()
        v_norm = float(torch.norm(v, p=2))
        if v_norm > tolerance:
            return -self._norm_radius * v.div(v_norm)  # Projection to Ball
        else:
            return torch.zeros_like(x)

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the KSupportNormBall w/ radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list
        x_norm = self.k_support_norm(torch.cat([p.flatten() for p in x]))
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        return x

    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for KSupportNormBall.")

    @torch.no_grad()
    def reset_k(self, k):
        self.k = min(k, self.n)

    @torch.no_grad()
    def k_support_norm(self, x, tol=1e-7):
        """Computes the k-support-norm of x"""
        sorted_increasing = torch.sort(torch.abs(x.flatten()), descending=False).values
        running_mean = torch.cumsum(sorted_increasing, 0)  # Compute the entire running_mean since this is optimized
        running_mean = running_mean[-self.k:]  # Throw away everything but the last entries k entries
        running_mean = running_mean / torch.arange(1, self.k + 1, device=x.device)
        lower = sorted_increasing[-self.k:]
        upper = torch.cat([sorted_increasing[-(self.k - 1):], torch.tensor([float('inf')], device=x.device)])
        relevantIndices = torch.nonzero(torch.logical_and(upper + tol > running_mean, running_mean + tol >= lower))[0]
        r = int(relevantIndices[0])  # Should have only one element, otherwise its a numerical problem -> pick first

        # With r, we can now compute the norm
        d = x.numel()
        x_right = 1 / (r + 1) * torch.sum(sorted_increasing[:d - (self.k - r) + 1]).pow(2)
        x_left = torch.sum(sorted_increasing[-(self.k - 1 - r):].pow(2)) if r < self.k - 1 else 0
        x_norm = torch.sqrt(x_left + x_right)
        return x_norm


class GroupKSupportNormBall(Constraint):
    """
    Assumes filters for now, i.e. dimension 1 of the Conv2d weight matrix
    """

    def __init__(self, n, k=1, l2_diameter=None, norm_radius=None):
        self.n_groups = n
        self.k = min(k, self.n_groups)

        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2.0 * norm_radius
        elif norm_radius is None:
            self._norm_radius = l2_diameter / 2.0
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in GroupKSupportNormBall w/ radius r minimizing v*x. Note: x is a list of params"""
        # Compute norm of each filter in each gradient tensor
        filter_norms = [torch.norm(d_p.flatten(start_dim=1), p=2, dim=1) for d_p in x]
        threshold = torch.kthvalue(torch.cat([d_p_norm.flatten() for d_p_norm in filter_norms]),
                                   k=self.n_groups - self.k + 1).values
        v_list = [torch.zeros_like(d_p) for d_p in x]
        cum_norm = 0
        for idx, d_p in enumerate(x):
            filterNormSelector = filter_norms[idx] >= threshold
            v_list[idx][filterNormSelector, :, :, :] = d_p[filterNormSelector, :, :, :]
            cum_norm += torch.sum(filter_norms[idx][filterNormSelector] ** 2)
        v_norm = torch.sqrt(cum_norm)
        if v_norm > tolerance:
            return [-self._norm_radius * v_i.div(v_norm) for v_i in v_list]
        else:
            return [torch.zeros_like(d_p) for d_p in x]

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the GroupKSupportNormBall w/ radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list
        x_norm = self.group_k_support_norm(x)
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        return x

    @torch.no_grad()
    def group_k_support_norm(self, x):
        """Computes the group-k-support-norm of x"""
        assert type(x) == list
        filter_norms = [torch.norm(p.flatten(start_dim=1), p=2, dim=1) for p in x]
        norm_vector = torch.cat(filter_norms)
        return self.k_support_norm(norm_vector)

    @torch.no_grad()
    def k_support_norm(self, x, tol=1e-10):
        """Computes the k-support-norm of x"""
        sorted_increasing = torch.sort(torch.abs(x.flatten()), descending=False).values
        running_mean = torch.cumsum(sorted_increasing, 0)  # Compute the entire running_mean since this is optimized
        running_mean = running_mean[-self.k:]  # Throw away everything but the last entries k entries
        running_mean = running_mean / torch.arange(1, self.k + 1, device=x.device)
        lower = sorted_increasing[-self.k:]
        upper = torch.cat([sorted_increasing[-(self.k - 1):], torch.tensor([float('inf')], device=x.device)])
        relevantIndices = torch.nonzero(torch.logical_and(upper + tol > running_mean, running_mean + tol >= lower))[0]
        r = int(relevantIndices[0])  # Should have only one element, otherwise its a numerical problem -> pick first

        # With r, we can now compute the norm
        d = x.numel()
        x_right = 1 / (r + 1) * torch.sum(sorted_increasing[:d - (self.k - r) + 1]).pow(2)
        x_left = torch.sum(sorted_increasing[-(self.k - 1 - r):].pow(2)) if r < self.k - 1 else 0
        x_norm = torch.sqrt(x_left + x_right)
        return x_norm

    @torch.no_grad()
    def euclidean_project(self, x):
        raise NotImplementedError(f"Projection not implemented for GroupKSupportNormBall.")


class NuclearNormBall(Constraint):
    def __init__(self, n, lmo_nuc_method, l2_diameter=None, norm_radius=None):
        self.n = n
        # print(lmo_nuc_method)
        if lmo_nuc_method == 'power_it':
            self.lmo_nuc_method = Utils.SVD_power_iteration
        elif lmo_nuc_method == 'eigval':
            self.lmo_nuc_method = Utils.SVD_eigval
        else:
            raise NotImplementedError(
                f"lmo_nuc_method {lmo_nuc_method} does not exist or is not suited for this constraint.")

        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2.0 * norm_radius
        elif norm_radius is None:
            self._norm_radius = l2_diameter / 2.0
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in NuclearNormBall w/ radius r minimizing v*x. Note: x is a list of params"""
        # Compute norm of each filter in each gradient tensor
        d_p = x[0]
        # u, v, sigma = Utils.SVD_power_iteration(-d_p.flatten(start_dim=1))
        u, v, sigma = self.lmo_nuc_method(-d_p.flatten(start_dim=1))  # No need to divide by sigma!
        if sigma > tolerance:
            return [self._norm_radius * torch.outer(u, v).view(d_p.shape)]
        else:
            return [torch.zeros_like(d_p)]

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the NuclearNormBall w/ radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list and len(x) == 1
        x_norm = self.nuclear_norm(x)
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        return x

    @torch.no_grad()
    def nuclear_norm(self, x):
        """Computes the nuclear_norm of x"""
        assert type(x) == list and len(x) == 1
        if len(x[0].shape) == 4:
            return torch.linalg.norm(x[0].flatten(start_dim=1), ord='nuc')
        return torch.linalg.norm(x[0], ord='nuc')

    @torch.no_grad()
    def euclidean_project(self, x):
        raise NotImplementedError(f"Projection not implemented for NuclearNormBall.")


class SpectralKSparsePolytope(Constraint):
    # Case p = inf
    def __init__(self, n, k, lmo_nuc_method, l2_diameter=None, norm_radius=None):
        self.n = n
        self.k = min(k, n)

        if lmo_nuc_method == 'partial':
            self.lmo_nuc_method = lambda W: Utils.SVD_partial(W, k=self.k)
        else:
            raise NotImplementedError(
                f"lmo_nuc_method {lmo_nuc_method} does not exist or is not suited for this constraint.")

        # Same conversion as for KSparsePolytope, since we just calculate with SVDvals
        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2.0 * norm_radius * math.sqrt(self.k)
        elif norm_radius is None:
            self._norm_radius = l2_diameter / (2.0 * math.sqrt(self.k))
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in SpectralKSparsePolytope w/ radius r minimizing v*x. Note: x is a list of params"""
        d_p = x[0]
        U, V_t, S = self.lmo_nuc_method(-d_p.flatten(start_dim=1))  # S not needed since U*V_t has nuclear norm = 1
        return [self._norm_radius * torch.mm(U, V_t).view(d_p.shape)]

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the SpectralKSparsePolytope w/ radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list and len(x) == 1
        x_norm = self.spectral_k_sparse_norm(x)
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        return x

    @torch.no_grad()
    def spectral_k_sparse_norm(self, x):
        """Computes the spectral_k_sparse_norm of x"""
        assert type(x) == list and len(x) == 1
        W = x[0]
        if len(W.shape) == 4:
            W = W.flatten(start_dim=1)
        svdVals = torch.linalg.svdvals(W)
        Linf = torch.norm(svdVals, p=float('inf'))
        L1k = torch.norm(svdVals / self.k, p=1)
        return max(Linf, L1k)

    @torch.no_grad()
    def euclidean_project(self, x):
        raise NotImplementedError(f"Projection not implemented for SpectralKSparsePolytope.")


class SpectralKSupportNormBall(Constraint):
    # Case p = 2
    def __init__(self, n, k, lmo_nuc_method, l2_diameter=None, norm_radius=None):
        self.n = n
        self.k = min(k, n)

        if lmo_nuc_method == 'partial':
            self.lmo_nuc_method = lambda W: Utils.SVD_partial(W, k=self.k)
        elif lmo_nuc_method == 'qrpartial':
            self.lmo_nuc_method = lambda W: Utils.SVD_block_power_iteration(W, k=self.k)
        else:
            raise NotImplementedError(
                f"lmo_nuc_method {lmo_nuc_method} does not exist or is not suited for this constraint.")

        # Same conversion as for KSparsePolytope, since we just calculate with SVDvals
        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2.0 * norm_radius
        elif norm_radius is None:
            self._norm_radius = l2_diameter / 2.0
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

    @torch.no_grad()
    def lmo(self, x):
        """Returns v in SpectralKSupportNormBall w/ radius r minimizing v*x. Note: x is a list of params"""
        d_p = x[0]
        U, V_t, S = self.lmo_nuc_method(-d_p.flatten(start_dim=1))
        # We can normalize here with L2-norm of S, since S contains only k entries anyway
        return [self._norm_radius / torch.norm(S, p=2) * torch.mm(U, torch.mm(torch.diag(S), V_t)).view(d_p.shape)]

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the SpectralKSupportNormBall w/ radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list and len(x) == 1
        x_norm = self.spectral_k_support_norm(x)
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        return x

    @torch.no_grad()
    def spectral_k_support_norm(self, x):
        """Computes the spectral_k_support_norm of x by computing the k_support_norm of sigma(x)"""
        assert type(x) == list and len(x) == 1
        W = x[0]
        if len(W.shape) == 4:
            W = W.flatten(start_dim=1)
        svdVals = torch.linalg.svdvals(W)
        return self.k_support_norm(svdVals)

    @torch.no_grad()
    def k_support_norm(self, x, tol=1e-10):
        """Computes the k-support-norm of x"""
        sorted_increasing = torch.sort(torch.abs(x.flatten()), descending=False).values
        running_mean = torch.cumsum(sorted_increasing, 0)  # Compute the entire running_mean since this is optimized
        running_mean = running_mean[-self.k:]  # Throw away everything but the last entries k entries
        running_mean = running_mean / torch.arange(1, self.k + 1, device=x.device)
        lower = sorted_increasing[-self.k:]
        upper = torch.cat([sorted_increasing[-(self.k - 1):], torch.tensor([float('inf')], device=x.device)])
        relevantIndices = torch.nonzero(torch.logical_and(upper + tol > running_mean, running_mean + tol >= lower))[0]
        r = int(relevantIndices[0])  # Should have only one element, otherwise its a numerical problem -> pick first

        # With r, we can now compute the norm
        d = x.numel()
        x_right = 1 / (r + 1) * torch.sum(sorted_increasing[:d - (self.k - r) + 1]).pow(2)
        x_left = torch.sum(sorted_increasing[-(self.k - 1 - r):].pow(2)) if r < self.k - 1 else 0
        x_norm = torch.sqrt(x_left + x_right)
        return x_norm

    @torch.no_grad()
    def euclidean_project(self, x):
        raise NotImplementedError(f"Projection not implemented for SpectralKSupportNormBall.")


class KSparsePolytope(Constraint):
    """
    # Polytopes with vertices v \in {0, +/- r}^n such that exactly k entries are nonzero
    # This is exactly the intersection of B_1(r*k) with B_inf(r)
    """

    def __init__(self, n, k=1, l2_diameter=None, norm_radius=None, adjust_diameter=False):
        super().__init__(n)
        self.k = min(k, n)

        if l2_diameter is None and norm_radius is None:
            raise ValueError("Neither diameter nor radius given")
        elif l2_diameter is None:
            self._norm_radius = norm_radius
            self._l2_diameter = 2.0 * norm_radius * math.sqrt(self.k)
        elif norm_radius is None:
            self._norm_radius = l2_diameter / (2.0 * math.sqrt(self.k))
            self._l2_diameter = l2_diameter
        else:
            raise ValueError("Both diameter and radius given")

        if adjust_diameter:
            # The l2_diameter stays the same, but we adjust the radius to allow increasing k to n
            self._norm_radius = math.sqrt(float(k) / n) * self._norm_radius
            self.k = n

    @torch.no_grad()
    def lmo(self, x):
        """Computes and formats single_lmo solutions"""
        # Apply LMO
        v = self.single_lmo(torch.cat([g.flatten() for g in x]))
        v_list = []

        # Update parameters
        seen_elements = 0
        for p in x:
            n_p = p.numel()
            v_list.append(v[seen_elements:seen_elements + n_p].view(p.shape))
            seen_elements += n_p
        return v_list

    @torch.no_grad()
    def single_lmo(self, x):
        """Returns v in KSparsePolytope w/ radius r minimizing v*x"""
        super().lmo(x)
        v = torch.zeros_like(x)
        maxIndices = torch.topk(torch.abs(x.flatten()), k=self.k).indices
        v.flatten()[maxIndices] = -self._norm_radius * torch.sign(x.flatten()[maxIndices])
        return v

    @torch.no_grad()
    def shift_inside(self, x):
        """Projects x to the KSparsePolytope with radius r.
        NOTE: This is a valid projection, although not the one mapping to minimum distance points.
        """
        assert type(x) == list
        x_norm = self.k_sparse_norm(torch.cat([p.flatten() for p in x]))
        if x_norm > self._norm_radius:
            return [self._norm_radius * p.div(x_norm) for p in x]
        return x

    @torch.no_grad()
    def euclidean_project(self, x):
        super().euclidean_project(x)
        raise NotImplementedError(f"Projection not implemented for K-sparse polytope.")

    @torch.no_grad()
    def reset_k(self, k):
        # This is based on the assumption that mode == 'initialization', i.e. the L2-diameter is specified
        self.k = min(k, self.n)

        # _l2_diameter stays the same
        self._norm_radius = self._l2_diameter / (2.0 * math.sqrt(self.k))

    @torch.no_grad()
    def k_sparse_norm(self, x):
        """Computes the k-sparse-norm of x"""
        Linf = torch.norm(x, p=float('inf'))
        L1k = torch.norm(x / self.k, p=1)
        return max(Linf, L1k)
