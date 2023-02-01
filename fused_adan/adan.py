import torch

import fused_adan

fused_adan_cuda = None

class Adan(torch.optim.Optimizer):
    """
    Modified from Fairseq and Use LightSeq adan kernel.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in LSAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        eps_inside_sqrt=False,
        weight_decay=0.0,
        max_grad_norm=0.0,
        amsgrad=False,
    ):
        global fused_adan_cuda

        if fused_adan_cuda is None:
            fused_adan_cuda = fused_adan
        if amsgrad:
            raise RuntimeError("LSAdam does not support the AMSGrad variant.")
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
        }
        super().__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    @property
    def supports_step_with_scale(self):
        return True

    def step(self, closure=None, grads=None, scale=1.0, grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        for group, grads_this_group, grad_norm in zip(
            self.param_groups, grads_group, grad_norms
        ):
            if grads_this_group is None:
                grads_this_group = [None] * len(group["params"])

            # compute combined scale factor for this group
            combined_scale = scale
            if group.get("max_grad_norm", 0) > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group["max_grad_norm"]
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group.get("bias_correction", 1) else 0

            for p, grad in zip(group["params"], grads_this_group):
                # note: p.grad should not ever be set for correct
                # operation of mixed precision optimizer that sometimes
                # sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "LSAdam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                out_p = p.data
                with torch.cuda.device(p.device):
                    fused_adan_cuda.adan(
                        p_data_fp32,
                        out_p,
                        exp_avg,
                        exp_avg_sq,
                        grad,
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        combined_scale,
                        state["step"],
                        self.eps_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss
    
import fused_adan

fused_adan_cuda = None
import math
from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for
        Faster Optimizing Deep Models[J].arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay
            (default: False)
        foreach (bool): if True would use torch._foreach implementation.
            It's faster but uses slightly more memory. (default: True)
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.98, 0.92, 0.99),
                 eps=1e-8,
                 weight_decay=0.0,
                 max_grad_norm=0.0,
                 no_prox=False,
                 foreach: bool = True):
        if not 0.0 <= max_grad_norm:
            raise ValueError('Invalid Max grad norm: {}'.format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(
                betas[2]))
        global fused_adan_cuda

        if fused_adan_cuda is None:
            fused_adan_cuda = fused_adan
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm,
                        no_prox=no_prox,
                        foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None, grads=None, scale=1.0, grad_norms=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'],
                                         device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(
                max_grad_norm / (global_grad_norm + group['eps']),
                max=1.0).item()
        else:
            clip_global_grad_norm = 1.0

        for group, grads_this_group, grad_norm in zip(
            self.param_groups, grads_group, grad_norms
        ):
            if grads_this_group is None:
                grads_this_group = [None] * len(group["params"])

             # compute combined scale factor for this group
            combined_scale = scale
            if group.get("max_grad_norm", 0) > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group["max_grad_norm"]
                if clip > 1:
                    combined_scale = clip * scale

            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support
            # by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            bias_correction3 = 1.0 - beta3**group['step']

            for p, grad in zip(group["params"], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                    
                p_data_fp32 = p.data.float()

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_diff'] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    state["exp_avg_diff"] = state["exp_avg_diff"].to(p_data_fp32)

                if 'pre_grad' not in state or group['step'] == 1:
                    # at first step grad wouldn't be clipped by `clip_global_grad_norm`
                    # this is only to simplify implementation
                    state['pre_grad'] = p.grad

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_diff = state["exp_avg_diff"]
                pre_grad = state["pre_grad"]
                
                out_p = p.data
                kwargs = dict(
                    params=p_data_fp32, // p            at::Tensor
                    params_copy=out_p,  // p_copy       at::Tensor
                    grad=grad,          // g            at::Tensor
                    exp_avg=exp_avg,    // exp_avg      at::Tensor            
                    exp_avg_sq=exp_avg_sq,// exp_avg_sq at::Tensor
                    exp_avg_diff=exp_avg_diff,// diff   at::Tensor
                    pre_grad=pre_grad,  // pre_g        at::Tensor
                    beta1=beta1,        // beta1        float
                    beta2=beta2,        // beta2        float  
                    beta3=beta3,        // beta3        float
                    bias_correction1=bias_correction1,  // bias_correction1 float
                    bias_correction2=bias_correction2,  // bias_correction2 float
                    bias_correction3_sqrt=math.sqrt(bias_correction3),  // bias_correction3_sqrt float
                    lr=group['lr'],     // lr           float
                    weight_decay=group['weight_decay'], // decay float
                    eps=group['eps'],   // eps          float
                    no_prox=group['no_prox'], // no_prox bool
                    scale=combined_scale,// grad_scale  float
                )

                with torch.cuda.device(p.device):
                    fused_adan_cuda.adan(**kwargs)
            
        return loss
