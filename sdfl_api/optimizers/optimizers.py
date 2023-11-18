import torch
import torch.optim as native_optimizers
from torchmetrics import MeanMetric


class SFW(native_optimizers.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of lr rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, lr=0.1, rescale='diameter', momentum=0, dampening=0, extensive_metrics=False,
                 device='cpu'):
        momentum = momentum or 0
        dampening = dampening or 0
        if rescale is None and not (0.0 <= lr <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"Momentum must be between 0 and 1, got {momentum} of type {type(momentum)}.")
        if not (0.0 <= dampening <= 1.0):
            raise ValueError("fDampening must be between 0 and 1, got {dampening} of type {type(dampening)}.")

        if rescale == 'None':
            rescale = None
        if not (rescale in ['diameter', 'gradient', 'fast_gradient', None]):
            raise ValueError(
                f"Rescale type must be either 'diameter', 'gradient', 'fast_gradient' or None, got {rescale} of type {type(rescale)}.")

        self.rescale = rescale
        self.extensive_metrics = extensive_metrics

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening)
        super(SFW, self).__init__(params, defaults)

        # Metrics
        self.metrics = {
            'grad_norm': MeanMetric().to(device=device),
            'grad_normalizer_norm': MeanMetric().to(device=device),
            'diameter_normalizer': MeanMetric().to(device=device),
            'effective_lr': MeanMetric().to(device=device),
        }

    def reset_metrics(self):
        for m in self.metrics.values():
            m.reset()

    def get_metrics(self):
        return {m: self.metrics[m].compute() if self.extensive_metrics else {} for m in self.metrics.keys()}

    @torch.no_grad()
    def reset_momentum(self):
        """Resets momentum, typically used directly after pruning"""
        for group in self.param_groups:
            momentum = group['momentum']
            if momentum > 0:
                for p in group['params']:
                    param_state = self.state[p]
                    if 'momentum_buffer' in param_state:
                        del param_state['momentum_buffer']

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        import time
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            momentum_buffer_list = []
            constraint = group['constraint']
            momentum = group['momentum']
            dampening = group['dampening']
            # Add momentum, fill grad list with momentum_buffers and concatenate
            grad_list = []
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = param_state['momentum_buffer']
                grad_list.append(d_p)

            # print(grad_list)
            # LMO solution
            v_list = constraint.lmo(grad_list)  # LMO optimal solution

            # Determine learning rate rescaling factor
            # s1 = time.time()
            factor = 1
            if self.rescale == 'diameter':
                # Rescale lr by diameter
                factor = 1. / constraint.get_diameter()
                if self.extensive_metrics:
                    self.metrics['grad_norm'](torch.norm(torch.cat([g.flatten() for g in grad_list]), p=2))
                    self.metrics['diameter_normalizer'](constraint.get_diameter())
            elif self.rescale in ['fast_gradient', 'gradient']:
                # Rescale lr by gradient
                # s11 = time.time()
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grad_list]), p=2) # L2 of gradients
                # print('s11', time.time()-s11)
                if self.rescale == 'fast_gradient':
                    # print(constraint.get_diameter())
                    # s12 = time.time()
                    grad_normalizer_norm = 0.5 * constraint.get_diameter() # 0.5 * (2.0 * value * L2 of params)
                    # print('s12', time.time() - s12)
                else:
                    grad_normalizer_norm = torch.norm(torch.cat([p.flatten()
                                                                 for p in group['params'] if
                                                                 p.grad is not None]) - torch.cat(
                        [v_i.flatten() for v_i in v_list]), p=2)

                factor = grad_norm / grad_normalizer_norm # L2 of gradient / L2 of params
                # s13 = time.time()
                if self.extensive_metrics:
                    self.metrics['grad_norm'](grad_norm)
                    self.metrics['grad_normalizer_norm'](grad_normalizer_norm)
                    self.metrics['diameter_normalizer'](constraint.get_diameter())
                # print('s13', time.time() - s13)

            # print(factor, group['lr'])
            # print('s1:',time.time()-s1)
            lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]
            # Update parameters
            # s2 = time.time()
            for p_idx, p in enumerate(group['params']):
                p.mul_(1 - lr)
                p.add_(v_list[p_idx], alpha=lr)
            # print('s2:', time.time() - s2)

            # for p_idx, p in enumerate(group['params']):
                # if momentum != 0:
                #     buf = momentum_buffer_list[p_idx]
                #
                #     if buf is None:
                #         buf = torch.clone(v_list[p_idx]).detach()
                #         momentum_buffer_list[p_idx] = buf
                #     else:
                #         buf.mul_(momentum).add_(v_list[p_idx], alpha=1 - dampening)

                # p.add_(v_list[p_idx], alpha=-lr)

            # for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            #     state = self.state[p]
            #     state['momentum_buffer'] = momentum_buffer

            if self.extensive_metrics:
                self.metrics['effective_lr'](lr)
        return loss

