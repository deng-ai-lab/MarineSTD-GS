import numpy as np
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.

    This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
    based on the number of elements in `weights`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

    Args:
        weights (Tensor): A 1D tensor of weights for each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement. Default is True.

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """
    num_elements = weights.size(0)

    if num_elements <= 2**24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(weights, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = weights / weights.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(
            num_elements, size=n, p=weights_np, replace=replacement
        )
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        optimizer = optimizers[name]
        for i, param_group in enumerate(optimizer.param_groups):
            p = param_group["params"][0]
            p_state = optimizer.state[p]
            del optimizer.state[p]
            for key in p_state.keys():
                if key != "step":
                    v = p_state[key]
                    p_state[key] = optimizer_fn(key, v)
            p_new = param_fn(name, p)
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = p_state
            params[name] = p_new


@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]))

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    revised_opacity: bool = False,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    # # # =============================== Unpaired Settting ========================================================================================
    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    # # # =============================== Unpaired Settting ========================================================================================

    # # 从 [N, 2C] 中提取 -> 分割成两半 -> 相加融合
    # def split_feature(tensor_2c):  # 输入是 [N, 2C]
    #     C = tensor_2c.shape[1] // 2
    #     return tensor_2c[:, :C] + tensor_2c[:, C:]

    # scales = torch.exp(split_feature(params["scales"][sel]))   # → [N, C]
    # quats = F.normalize(split_feature(params["quats"][sel]), dim=-1)
    # # # =============================== Unpaired Settting ========================================================================================


    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    # # # # =============================== Unpaired Settting ========================================================================================
    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
        elif name == "scales":
            p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new)
        return p_new
    # # # =============================== Unpaired Settting ========================================================================================
    # def param_fn(name: str, p: Tensor) -> Tensor:
    #     repeats = [2] + [1] * (p.dim() - 1)
    #     C = p.shape[1] // 2  # 特征维度减半
    #     if name == "means":
    #         means_primary = p[:, :C]       # [N, C]
    #         means_secondary = p[:, C:]     # [N, C]
    #         # 只对 primary 部分进行偏移，samples 是 [2, N, C]
    #         moved_means = (means_primary[sel] + samples).reshape(-1, C)  # [2N, C]
    #         # 重复 secondary（不偏移），拼接为 [2N, 2C]
    #         repeated_secondary = torch.full_like(moved_means, 0)  # [2N, C]     
    #         p_split = torch.cat([moved_means, repeated_secondary], dim=1)  # [2N, 2C]
                    
    #     elif name == "scales":
    #         scale_primary = torch.log(scales / 1.6).repeat(2, 1)
    #         # 构造接近 0 的初始化值，建议使用 torch.log(torch.full(...)) 保持数值范围一致
    #         scale_secondary = torch.full_like(scale_primary,0)  # [2N, C]
    #         # 拼接为 [2N, 2C]
    #         p_split = torch.cat([scale_primary, scale_secondary], dim=1)            
            
    #     elif name == "opacities" and revised_opacity:
    #         p_combined = p[:, :C] + p[:, C:]  # [N, C]
    #         new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p_combined[sel]))  # [M, C]
    #         primary = torch.logit(new_opacities)  # [M, C]
    #         # 构造后半部分为较小值，例如 0.01 的 logit ≈ -4.6
    #         secondary = torch.full_like(primary, 0)  # [2N, C]
    #         p_split = torch.cat([primary, secondary], dim=1)  # [M, 2C]
    #     else:
    #         p_split = p[sel].repeat(repeats)  # 维持原有维度 [2N, 2C]
    #     p_new = torch.cat([p[rest], p_split])
    #     return torch.nn.Parameter(p_new)
    # # # # =============================== Unpaired Settting ========================================================================================


    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


@torch.no_grad()
def remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel])

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        value: The value to reset the opacities
    """

    # # =============================== Unpaired Settting ========================================================================================
    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            return torch.nn.Parameter(opacities)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")
    # # =============================== Unpaired Settting ========================================================================================
    # def param_fn(name: str, p: Tensor) -> torch.nn.Parameter:
    #     if name == "opacities":
    #         C = p.shape[1] // 2  # 处理 [N, 2C]
            
    #         # 合并主副通道进行 clamp
    #         p_combined = p[:, :C] + p[:, C:]
    #         max_logit = torch.logit(torch.tensor(value)).item()
            
    #         # 对叠加后的值进行限制
    #         clamped = torch.clamp(p_combined, max=max_logit)

    #         # 后半部分设为小常数
    #         secondary = torch.full_like(clamped,0)

    #         # 拼接得到新的参数
    #         new_opacity = torch.cat([clamped, secondary], dim=1)  # [N, 2C]

    #         return torch.nn.Parameter(new_opacity)

    #     else:
    #         raise ValueError(f"Unexpected parameter name: {name}")
    # # =============================== Unpaired Settting ========================================================================================        

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


@torch.no_grad()
def relocate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    """Inplace relocate some dead Gaussians to the lives ones.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to indicates which Gaussians are dead.
    """
    # support "opacities" with shape [N,] or [N, 1]
    opacities = torch.sigmoid(params["opacities"])

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)

    # Sample for new GSs
    eps = torch.finfo(torch.float32).eps
    probs = opacities[alive_indices].flatten()  # ensure its shape is [N,]
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0


@torch.no_grad()
def sample_add(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    n: int,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = opacities.flatten()
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p = torch.cat([p, p[sampled_idxs]])
        return torch.nn.Parameter(p)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v_new))


@torch.no_grad()
def inject_noise_to_position(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    scaler: float,
):
    opacities = torch.sigmoid(params["opacities"].flatten())
    scales = torch.exp(params["scales"])
    covars, _ = quat_scale_to_covar_preci(
        params["quats"],
        scales,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )

    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    noise = (
        torch.randn_like(params["means"])
        * (op_sigmoid(1 - opacities)).unsqueeze(-1)
        * scaler
    )
    noise = torch.einsum("bij,bj->bi", covars, noise)
    params["means"].add_(noise)
