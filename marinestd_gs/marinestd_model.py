from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from pytorch_msssim import SSIM
from torch.nn import Parameter
from torchmetrics.functional.regression import pearson_corrcoef

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

try:
    from gsplat.rendering import ns_spherical_harmonics
except ImportError:
    print("Please install customized gsplat version with ns_spherical_harmonics support")

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

from marinestd_gs.utils import (
    InstantaneousBrightnessFeatureEncoder,
    WaterParametersExtractor,
    adaptive_edge_aware_depth_smoothness_loss,
    negative_perturbation_regularization,
    project_and_sample_local_feature,
)

from marinestd_gs.math import k_nearest_sklearn, random_quat_tensor
from marinestd_gs.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases



def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat




@dataclass
class MarineSTDGsModelConfig(ModelConfig):
    """MarineSTD-GS model config."""

    _target: Type = field(default_factory=lambda: MarineSTDGsModel)
    
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    
    strategy: Literal["default", "mcmc"] = "default"
    """The default strategy will be used if strategy is not specified. Other strategies, e.g. mcmc, can be used."""
    max_gs_num: int = 1_000_000
    """Maximum number of GSs. Default to 1_000_000."""
    noise_lr: float = 5e5
    """MCMC samping noise learning rate. Default to 5e5."""
    mcmc_opacity_reg: float = 0.01
    """Regularization term for opacity in MCMC strategy. Only enabled when using MCMC strategy"""
    mcmc_scale_reg: float = 0.01
    """Regularization term for scale in MCMC strategy. Only enabled when using MCMC strategy"""

    # TD-branch settings
    enable_TD_branch: bool = True
    """Whether to enable the TD branch for transient illumination perturbation modeling."""
    enable_perturbation_relu: bool = False
    """Whether to force the additive illumination perturbation to be non-negative."""
    enable_negative_perturbation_regularization: bool = False
    """Whether to penalize negative additive illumination perturbations."""
    perturbation_regularization_weight: float = 100
    """Regularization weight for additive illumination perturbations."""
    negative_perturbation_regularization_weight: float = 10000
    """Regularization weight for negative additive illumination perturbations."""
    
    # SD-branch settings
    enable_SD_branch: bool = True
    """Whether to enable the underwater image formation branch."""
    enable_medium_bg: bool = True
    """Whether to use the global background medium term."""
    # Depth losses
    enable_coarse_depth_supervision: bool = True
    """Whether to enable the Coarse Depth Supervision term L_cds."""
    enable_adaptive_depth_smoothness: bool = True
    """Whether to enable the Adaptive Edge-aware Depth Smoothness term L_ads."""
    weight_ads: float = 0.01
    """Loss weight for the Adaptive Edge-aware Depth Smoothness term L_ads."""
    weight_cds: float = 0.1
    """Loss weight for the Coarse Depth Supervision term L_cds."""    

    # Multi-stage optimization schedule.
    stage_1: int = 10000
    """Step at which Stage I ends and Stage II begins."""
    stage_2: int = 20000
    """Step at which Stage II ends and Stage III begins."""
    
    
    # Additional regularization and refinement controls for the MarineSTD-GS training setup.
    enable_reg_loss: bool = False
    """Whether to enable the additional regularization loss term."""
    reset_alpha_value: float = -1.0
    """Opacity reset value applied after densification when a positive value is provided."""
    cull_alpha_thresh_post: float = -1.0
    """Post-densification opacity threshold used for pruning when a positive value is provided."""
    continue_cull_post_densification: bool = False
    """Whether to continue opacity-based pruning after the densification stage ends."""

class MarineSTDGsModel(Model):
    """MarineSTD-GS model.

    Derived from Nerfstudio's Splatfacto and extended with underwater
    spatiotemporal degradation modeling.

    Args:
        config: MarineSTD-GS model configuration
    """

    config: MarineSTDGsModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.total_image_num = kwargs['metadata']['total_image_num']  # Total number of training images.
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        
        # ------------------------------------------------------------- TD branch initialization ------------------------------
        # Scene contraction
        self.spatial_distortion = SceneContraction(order=float("inf"))

        # Fixed internal configuration for the hash-based MLP encoder.
        hash_num_levels = 16
        hash_min_res = 16
        hash_max_res = 8192
        hash_log2_hashmap_size = 21
        hash_features_per_level = 2

        # Register buffers
        self.register_buffer("aabb", self.scene_box.aabb)
        self.register_buffer("max_res", torch.tensor(hash_max_res))
        self.register_buffer("num_levels", torch.tensor(hash_num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(hash_log2_hashmap_size))
        
        # Intrinsic position encoder phi(.) in the paper, implemented with HashEncoding.
        # Its output is the intrinsic position feature f_mu.
        self.intrinsic_position_encoder = HashEncoding(
            num_levels=hash_num_levels,
            min_res=hash_min_res,
            max_res=hash_max_res,
            log2_hashmap_size=hash_log2_hashmap_size,
            features_per_level=hash_features_per_level,
            implementation="tcnn",
        )
        
        # Intrinsic Gaussian color encoder omega(.) in the paper.
        # Its output is the intrinsic color feature f_c.
        intrinsic_color_feature_dim = 32
        self.intrinsic_color_encoder = MLP(
            in_dim=3,
            num_layers=2,
            layer_width=16,
            out_dim=intrinsic_color_feature_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="tcnn",
        )

        # Instantaneous Brightness Feature Encoder in the paper.
        brightness_feature_dim = 16
        local_brightness_feature_dim = global_brightness_feature_dim = brightness_feature_dim
        self.instantaneous_brightness_encoder = InstantaneousBrightnessFeatureEncoder(
            input_channels=3,
            base_channels=brightness_feature_dim
        )
        
        # Input feature dimension of the Illumination Perturbation Decoder.
        in_dim_illumination_decoder = local_brightness_feature_dim + global_brightness_feature_dim
        in_dim_illumination_decoder += self.intrinsic_position_encoder.get_out_dim()  # Intrinsic position feature f_mu
        in_dim_illumination_decoder += intrinsic_color_feature_dim  # Intrinsic color feature f_c
         
        # TD-branch Illumination Perturbation Decoder.
        # Predicts additive transient illumination perturbation for each Gaussian.        
        self.illumination_perturbation_decoder = MLP(
            in_dim=in_dim_illumination_decoder,
            num_layers=2,       # Number of layers in the Illumination Perturbation Decoder
            layer_width=128,    # Hidden dimension of the Illumination Perturbation Decoder
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="tcnn",
        ) 


        
        # Optional non-negative constraint for additive caustic illumination perturbations.
        self.caustic_perturbation_nonneg_constraint  = nn.ReLU()
        
        # ------------------------------------------------------------- SD branch initialization ------------------------------        
        self.ambient_light_activation = nn.Sigmoid()
        self.medium_coeff_activation = nn.Softplus()

        # Water Parameters Extractor (WPE) in the SD Prediction branch.
        self.water_parameters_extractor = WaterParametersExtractor(input_channels=3, hidden_dim=32, output_dim=9)
        
        # ------------------------------------------------------------- Temporary caches ------------------------------
        self.depth_cache = {}  # Keys are (image_idx, downscale_factor).
        self.gt_img_cache = {}  # Keys are (image_idx, downscale_factor).

        # ------------------------------------------------------------- 3DGS parameter initialization ------------------------------
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        distances, _ = k_nearest_sklearn(means.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):

            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:             
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        if self.config.strategy == "default":
            
            if self.config.reset_alpha_every < 10:
                pause_refine_after_reset = (
                    250 if self.num_train_data + self.config.refine_every > 250
                    else self.num_train_data + self.config.refine_every
                )
            else:
                pause_refine_after_reset = self.num_train_data + self.config.refine_every

            
            # Strategy for GS densification
            self.strategy = DefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=pause_refine_after_reset,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
                reset_alpha_value=self.config.reset_alpha_value,
                cull_alpha_thresh_post=self.config.cull_alpha_thresh_post,
                continue_cull_post_densification=self.config.continue_cull_post_densification,
                
            )
            print(f'reset_every is {self.config.reset_alpha_every * self.config.refine_every}')
            print(f'pause_refine_after_reset is {self.num_train_data + self.config.refine_every} and set {pause_refine_after_reset}')
            
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(
                f"MarineSTD-GS does not support strategy {self.config.strategy}. "
                "Currently, the supported strategies include default and mcmc."
            )

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    
    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]
    
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],  # the learning rate for the "means" attribute of the GS
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        
        gps["water_parameters_extractor"] = list(self.water_parameters_extractor.parameters())

        gps["instantaneous_brightness_encoder"] = list(self.instantaneous_brightness_encoder.parameters())
        gps["intrinsic_position_encoder"] = list(self.intrinsic_position_encoder.parameters())
        gps["intrinsic_color_encoder"] = list(self.intrinsic_color_encoder.parameters())
        gps["illumination_perturbation_decoder"] = list(self.illumination_perturbation_decoder.parameters())
        
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        ambient_light = background.new_zeros(1, 3)
        zeros_coeff = background.new_zeros(1, 3)
        illumination_perturbation = background.new_zeros(0, 3)
        return {
            "rgb": rgb,
            "rgb_spatial_degraded": rgb,
            "rgb_spatiotemporal_degraded": rgb,
            "rgb_spatiotemporal_degraded_for_sd_grad": rgb,
            "rgb_spatiotemporal_degraded_for_td_grad": rgb,
            "rgb_attenuation_map": rgb,
            "rgb_backscatter_map": rgb,
            "rgb_caustic_pattern": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "background": background,
            "illumination_perturbation": illumination_perturbation,
            "ambient_light_image": rgb,
            "ambient_light": ambient_light,
            "attenuation_coeff": zeros_coeff,
            "backscatter_coeff": zeros_coeff,
        }

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        # make xy grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]

    @staticmethod
    def _compute_normalized_gs_distances(means_crop: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
        """Compute per-Gaussian camera distance used by the SD image formation model."""
        camtoworlds = torch.inverse(viewmat)  # [C, 4, 4]
        gs_directions = means_crop[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        gs_directions = gs_directions.detach()[0, :, :]  # [C, N, 3] -> [N, 3]
        gs_distances = torch.norm(gs_directions, dim=1, keepdim=True)  # [N, 1]
        return gs_distances / 10

    def get_render_outputs(
        self,
        render_config,
        camera_metadata,
        means_crop: torch.Tensor,
        quats_crop: torch.Tensor,
        scales_crop: torch.Tensor,
        opacities_crop: torch.Tensor,
        intrinsic_colors_crop: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        render_mode: str,
        sh_degree_to_use,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Render-only path for MarineSTD-GS rendering workflows.

        This function is entered only when ``camera.metadata["marine_render_config"]``
        is present. The normal training/evaluation path does not set that key and
        therefore does not use this function.

        Supported render usage patterns:

        1. Dataset render with per-frame metadata available
           - Typical for dataset-based rendering, where each camera corresponds to
             a real dataset image.
           - ``camera.metadata`` is expected to provide ``input_img``, ``depth_img``,
             and ``hard_image_id``.
           - ``render_config`` may be ``True`` or an empty dict to request the
             default render behavior.
           - TD and SD remain enabled by default, and ``water_param_save_dir`` may
             optionally be provided to save per-frame water parameters.

        2. Dataset render with water-parameter export
           - Same as case 1, but ``render_config["water_param_save_dir"]`` is set.
           - The estimated water parameters of each frame are saved as
             ``{hard_image_id}.pt``.

        3. Path render without per-frame image/depth inputs
           - Typical for camera-path, interpolated-path, or spiral rendering, where
             only the camera trajectory is available.
           - In this case, ``input_img``, ``depth_img``, and ``hard_image_id`` are
             normally unavailable.
           - TD must be disabled via ``render_config["disable_td"] = True``.
           - If SD is still needed, ``render_config["water_param_load_path"]`` must
             provide a precomputed water-parameter file; otherwise SD must also be
             disabled.
        """
        if render_config is True:
            render_config = {}
        camera_metadata = camera_metadata or {}
        input_img = camera_metadata.get("input_img", None)
        depth_img = camera_metadata.get("depth_img", None)
        hard_image_id = camera_metadata.get("hard_image_id", None)

        disable_td = render_config.get("disable_td", False)
        if self.config.enable_TD_branch and not disable_td:
            if input_img is None:
                raise ValueError("Render TD branch requires camera.metadata['input_img'].")
            positions = self.spatial_distortion(means_crop.detach())
            positions = (positions + 2.0) / 4.0
            intrinsic_position_feature = self.intrinsic_position_encoder(positions)
            intrinsic_color_feature = self.intrinsic_color_encoder(intrinsic_colors_crop.detach())
            brightness_features = self.instantaneous_brightness_encoder(input_img, device=means_crop.device)
            local_brightness_feature, _ = project_and_sample_local_feature(
                points_world=means_crop,
                viewmat=viewmat[0],
                K=K[0],
                feature_map=brightness_features["local_feature_map"],
                image_resolution=(H, W),
            )
            global_brightness_feature = brightness_features["global_feature"].unsqueeze(0).expand(means_crop.shape[0], -1)
            decoder_input = torch.cat(
                [
                    local_brightness_feature,
                    global_brightness_feature,
                    intrinsic_position_feature,
                    intrinsic_color_feature,
                ],
                dim=-1,
            )
            illumination_perturbation = self.illumination_perturbation_decoder(decoder_input)
            if self.config.enable_perturbation_relu:
                illumination_perturbation = self.caustic_perturbation_nonneg_constraint(illumination_perturbation)
        else:
            illumination_perturbation = torch.zeros((means_crop.shape[0], 3), device=means_crop.device)

        disable_sd = render_config.get("disable_sd", False)
        if self.config.enable_SD_branch and not disable_sd:
            # Render-time water-parameter IO is controlled only by marine_render_config:
            # - water_param_load_path: a single .pt file containing ambient/attenuation/backscatter.
            # - water_param_save_dir: a directory where current-frame parameters are saved as {hard_image_id}.pt.
            water_param_save_dir = render_config.get("water_param_save_dir", "")
            water_param_load_path = render_config.get("water_param_load_path", "")

            if water_param_load_path != "":
                water_param_cache = torch.load(water_param_load_path, map_location=means_crop.device)
                ambient_light = water_param_cache["ambient_light"].to(means_crop.device)
                attenuation_coeff = water_param_cache["attenuation_coeff"].to(means_crop.device)
                backscatter_coeff = water_param_cache["backscatter_coeff"].to(means_crop.device)
            else:
                if input_img is None or depth_img is None:
                    raise ValueError(
                        "Render SD branch requires camera.metadata['input_img'] and ['depth_img'] "
                        "unless 'water_param_load_path' is provided or 'disable_sd' is True."
                    )
                water_params = self.water_parameters_extractor(input_img, depth_img, device=means_crop.device)
                ambient_light = self.ambient_light_activation(water_params[..., :3])
                attenuation_coeff = self.medium_coeff_activation(water_params[..., 3:6])
                backscatter_coeff = self.medium_coeff_activation(water_params[..., 6:9])

                if water_param_save_dir != "":
                    if hard_image_id is None:
                        raise ValueError("Saving render water parameters requires camera.metadata['hard_image_id'].")
                    os.makedirs(water_param_save_dir, exist_ok=True)
                    water_param_save_path = os.path.join(water_param_save_dir, f"{hard_image_id}.pt")
                    torch.save(
                        {
                            "ambient_light": ambient_light.detach().cpu(),
                            "attenuation_coeff": attenuation_coeff.detach().cpu(),
                            "backscatter_coeff": backscatter_coeff.detach().cpu(),
                        },
                        water_param_save_path,
                    )

            ambient_light_image = ambient_light.view(1, 1, 3).expand(H, W, 3)
        else:
            ambient_light = torch.zeros((1, 3), device=means_crop.device)
            attenuation_coeff = torch.zeros((1, 3), device=means_crop.device)
            backscatter_coeff = torch.zeros((1, 3), device=means_crop.device)
            ambient_light_image = torch.zeros((H, W, 3), device=means_crop.device)

        normalized_gs_distances = self._compute_normalized_gs_distances(means_crop, viewmat)
        attenuation_map_colors_crop = torch.exp(-attenuation_coeff * normalized_gs_distances)
        backscatter_map_colors_crop = ambient_light * (1 - torch.exp(-backscatter_coeff * normalized_gs_distances))
        caustic_pattern_colors_crop = illumination_perturbation

        spatial_degraded_colors_crop = intrinsic_colors_crop * attenuation_map_colors_crop + backscatter_map_colors_crop
        spatiotemporal_degraded_colors_crop = spatial_degraded_colors_crop + illumination_perturbation

        render_colors_crop = torch.cat(
            (
                intrinsic_colors_crop,
                spatial_degraded_colors_crop,
                spatiotemporal_degraded_colors_crop,
                attenuation_map_colors_crop,
                backscatter_map_colors_crop,
                caustic_pattern_colors_crop,
            ),
            dim=-1,
        )

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=render_colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )

        alpha = alpha[:, ...]
        background = self._get_background_color()
        rgb = torch.clamp(render[:, ..., :3], 0.0, 1.0)
        rgb_spatial_degraded = render[:, ..., 3:6]
        rgb_spatiotemporal_degraded = render[:, ..., 6:9]
        rgb_attenuation_map = render[:, ..., 9:12]
        rgb_backscatter_map = render[:, ..., 12:15]
        rgb_caustic_pattern = render[:, ..., 15:18]

        if self.config.enable_medium_bg:
            rgb_spatial_degraded = rgb_spatial_degraded + (1 - alpha) * ambient_light_image.unsqueeze(0)
            rgb_spatiotemporal_degraded = rgb_spatiotemporal_degraded + (1 - alpha) * ambient_light_image.unsqueeze(0)

        depth_im = render[:, ..., -1:]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "rgb_spatial_degraded": rgb_spatial_degraded.squeeze(0),  # type: ignore
            "rgb_spatiotemporal_degraded": rgb_spatiotemporal_degraded.squeeze(0),  # type: ignore
            "rgb_attenuation_map": rgb_attenuation_map.squeeze(0),  # type: ignore
            "rgb_backscatter_map": rgb_backscatter_map.squeeze(0),  # type: ignore
            "rgb_caustic_pattern": rgb_caustic_pattern.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "illumination_perturbation": illumination_perturbation,  # type: ignore
            "ambient_light_image": ambient_light_image,  # type: ignore
            "ambient_light": ambient_light,
            "attenuation_coeff": attenuation_coeff,
            "backscatter_coeff": backscatter_coeff,
        }  # type: ignore

    def get_metadata_free_outputs(
        self,
        means_crop: torch.Tensor,
        quats_crop: torch.Tensor,
        scales_crop: torch.Tensor,
        opacities_crop: torch.Tensor,
        intrinsic_colors_crop: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        render_mode: str,
        sh_degree_to_use,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Fallback for ns-render cameras that do not carry MarineSTD-GS metadata."""
        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=intrinsic_colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )

        alpha = alpha[:, ...]
        background = self._get_background_color()
        rgb = torch.clamp(render[:, ..., :3], 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        zeros_image = torch.zeros((H, W, 3), device=means_crop.device)
        zeros_coeff = torch.zeros((1, 3), device=means_crop.device)
        illumination_perturbation = torch.zeros((means_crop.shape[0], 3), device=means_crop.device)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "rgb_spatial_degraded": rgb.squeeze(0),  # type: ignore
            "rgb_spatiotemporal_degraded": rgb.squeeze(0),  # type: ignore
            "rgb_spatiotemporal_degraded_for_sd_grad": rgb.squeeze(0),  # type: ignore
            "rgb_spatiotemporal_degraded_for_td_grad": rgb.squeeze(0),  # type: ignore
            "rgb_attenuation_map": zeros_image,  # type: ignore
            "rgb_backscatter_map": zeros_image,  # type: ignore
            "rgb_caustic_pattern": zeros_image,  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "illumination_perturbation": illumination_perturbation,  # type: ignore
            "ambient_light_image": zeros_image,  # type: ignore
            "ambient_light": zeros_coeff,
            "attenuation_coeff": zeros_coeff,
            "backscatter_coeff": zeros_coeff,
        }  # type: ignore

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]: # type: ignore
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color.to(self.device)
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        intrinsic_colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        depth_loss_enabled = self.config.enable_coarse_depth_supervision or self.config.enable_adaptive_depth_smoothness
        if self.config.output_depth_during_training or depth_loss_enabled or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"       

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            intrinsic_colors_crop = ns_spherical_harmonics(
                means_crop, intrinsic_colors_crop, viewmat, sh_degree=sh_degree_to_use
            )
            sh_degree_to_use = None            
        else:
            intrinsic_colors_crop = torch.sigmoid(intrinsic_colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        camera_metadata = getattr(camera, 'metadata', None)
        marine_render_config = None if camera_metadata is None else camera_metadata.get("marine_render_config", None)
        if marine_render_config is not None:
            return self.get_render_outputs(
                render_config=marine_render_config,
                camera_metadata=camera_metadata,
                means_crop=means_crop,
                quats_crop=quats_crop,
                scales_crop=scales_crop,
                opacities_crop=opacities_crop,
                intrinsic_colors_crop=intrinsic_colors_crop,
                viewmat=viewmat,
                K=K,
                W=W,
                H=H,
                render_mode="RGB+ED",
                sh_degree_to_use=sh_degree_to_use,
            )
        if camera_metadata is None:
            return self.get_metadata_free_outputs(
                means_crop=means_crop,
                quats_crop=quats_crop,
                scales_crop=scales_crop,
                opacities_crop=opacities_crop,
                intrinsic_colors_crop=intrinsic_colors_crop,
                viewmat=viewmat,
                K=K,
                W=W,
                H=H,
                render_mode=render_mode,
                sh_degree_to_use=sh_degree_to_use,
            )

        # Enable TD branch 
        if self.config.enable_TD_branch:
                            
            # Intrinsic position feature f_mu
            positions = means_crop.detach()
            positions = self.spatial_distortion(positions)
            
            # Normalize positions from [-2, 2] to [0, 1]
            positions = (positions + 2.0) / 4.0    # shape is [N,3]
            intrinsic_position_feature = self.intrinsic_position_encoder(positions)  # f_mu in the paper
            
            # intrinsic color feature f_c
            intrinsic_gaussian_color = intrinsic_colors_crop.detach()
            intrinsic_color_feature = self.intrinsic_color_encoder(intrinsic_gaussian_color)  # f_c in the paper
            
            # Instantaneous Brightness Feature Encoder: extract F_l and f_g from the degraded image.
            input_img = camera.metadata['input_img']                 # shape is [H,W,3]    
            brightness_features = self.instantaneous_brightness_encoder(input_img, device=means_crop.device)
            local_brightness_feature, _ = project_and_sample_local_feature(
                points_world=means_crop,
                viewmat=viewmat[0],  # [ Num_cam, 4,4 ]
                K=K[0],                 # [ Num_cam, 3,3 ]
                feature_map=brightness_features['local_feature_map'],
                image_resolution=(H, W)
            )

            global_brightness_feature = brightness_features['global_feature'].unsqueeze(0).expand(means_crop.shape[0], -1)

            decoder_input = torch.cat(
                [
                    local_brightness_feature,       # f_l in the paper
                    global_brightness_feature,      # f_g in the paper
                    intrinsic_position_feature,     # f_mu in the paper
                    intrinsic_color_feature,        # f_c in the paper
                ],
                dim=-1,
            )
            illumination_perturbation = self.illumination_perturbation_decoder(decoder_input)
            
            # Optional non-negative constraint for the additive perturbation.
            if self.config.enable_perturbation_relu:
                illumination_perturbation = self.caustic_perturbation_nonneg_constraint(illumination_perturbation)
        
        # Disable TD branch 
        else:
            illumination_perturbation = torch.zeros((means_crop.shape[0], 3), device=means_crop.device)
                
        
        # Enable SD branch 
        if self.config.enable_SD_branch:
            # Water-parameter cache/load behavior is render-only and lives in get_render_outputs().
            if camera_metadata is not None:
                input_img = camera_metadata.get('input_img', None)
                depth_img = camera_metadata.get('depth_img', None)
            else:
                input_img = None
                depth_img = None

            assert input_img is not None and depth_img is not None, \
                "Missing input/depth image for SD branch"
            water_params = self.water_parameters_extractor(input_img, depth_img, device=means_crop.device)
            ambient_light = self.ambient_light_activation(water_params[..., :3])
            attenuation_coeff = self.medium_coeff_activation(water_params[..., 3:6])
            backscatter_coeff = self.medium_coeff_activation(water_params[..., 6:9])
                
            ambient_light_image = ambient_light.view(1, 1, 3).expand(H, W, 3)

        # Disable SD branch 
        else:               
            ambient_light = torch.zeros((1, 3), device=means_crop.device )
            attenuation_coeff = torch.zeros((1, 3), device=means_crop.device )
            backscatter_coeff = torch.zeros((1, 3), device=means_crop.device )
            ambient_light_image = torch.zeros((H, W, 3), device=means_crop.device )
            
        # ---------------------------------------------------------------------- Render according to the current optimization stage. ----------------------------------------------------------------------
        if self.step < self.config.stage_1:
            # Stage I: decoupled warm-up for the SD and TD branches.
            # The SD path updates Gaussian appearance/geometry and water parameters,
            # while the TD path is rendered separately with the SD/Gaussian path detached.
                            
            # 1. SD path: compute per-Gaussian spatially degraded colors using the
            # underwater image formation model.
            normalized_gs_distances = self._compute_normalized_gs_distances(means_crop, viewmat)

            spatial_degraded_colors_crop = (
                intrinsic_colors_crop * torch.exp(-attenuation_coeff * normalized_gs_distances)
                + ambient_light * (1 - torch.exp(-backscatter_coeff * normalized_gs_distances))
            )
            intrinsic_spatial_colors_crop = torch.cat(
                (intrinsic_colors_crop, spatial_degraded_colors_crop), dim=-1
            )
            
            
            # 2. First rasterization: render intrinsic colors and SD-degraded colors.
            intrinsic_spatial_render, alpha, self.info = rasterization(
                means=means_crop,
                quats=quats_crop,  # rasterization does normalization internally
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=intrinsic_spatial_colors_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
            if self.training:
                self.strategy.step_pre_backward(
                    self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
                )

            alpha = alpha[:, ...]
            background = self._get_background_color()
            
            rgb = intrinsic_spatial_render[:, ..., :3]
            rgb = torch.clamp(rgb, 0.0, 1.0)
            
            rgb_spatial_degraded = intrinsic_spatial_render[:, ..., 3:6]
            
            if self.config.enable_medium_bg:
                rgb_spatial_degraded = rgb_spatial_degraded + (1 - alpha) * ambient_light_image.unsqueeze(0)
            
            if render_mode == "RGB+ED":
                depth_im = intrinsic_spatial_render[:, ..., -1:]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
            else:
                depth_im = None
            
            # 3. Second rasterization: detach SD/Gaussian quantities so only the TD
            # branch receives gradients from the spatiotemporal degraded image.
            spatiotemporal_degraded_colors_crop_for_td_grad = (
                spatial_degraded_colors_crop.detach().clone() + illumination_perturbation
            )

            safe_means = means_crop.detach().clone()
            safe_quats = quats_crop.detach().clone()
            safe_scales = torch.exp(scales_crop).detach().clone()
            safe_opacities = torch.sigmoid(opacities_crop).squeeze(-1).detach().clone()
            
            spatiotemporal_render_for_td_grad, _, _ = rasterization(
                means=safe_means,
                quats=safe_quats,
                scales=safe_scales,
                opacities=safe_opacities,
                colors=spatiotemporal_degraded_colors_crop_for_td_grad,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
            rgb_spatiotemporal_degraded_for_td_grad = spatiotemporal_render_for_td_grad[:, ..., :3]

            if self.config.enable_medium_bg:
                rgb_spatiotemporal_degraded_for_td_grad = (
                    rgb_spatiotemporal_degraded_for_td_grad
                    + (1 - alpha.detach()) * ambient_light_image.unsqueeze(0).detach()
                )
                
            return {
                "rgb": rgb.squeeze(0),  # type: ignore
                "rgb_spatial_degraded": rgb_spatial_degraded.squeeze(0),  # type: ignore
                "rgb_spatiotemporal_degraded_for_td_grad": rgb_spatiotemporal_degraded_for_td_grad.squeeze(0),  # type: ignore

                "depth": depth_im,  # type: ignore
                "accumulation": alpha.squeeze(0),  # type: ignore
                "background": background,  # type: ignore
                
                "illumination_perturbation": illumination_perturbation,  # type: ignore
                
                "ambient_light_image": ambient_light_image,  # type: ignore
                "ambient_light": ambient_light,
                "attenuation_coeff": attenuation_coeff,
                "backscatter_coeff": backscatter_coeff,
            }  # type: ignore


        elif self.step >= self.config.stage_1 and self.step < self.config.stage_2:
            # Stage II: interleaved decoupled optimization for SD and TD.
            # One spatiotemporal path updates SD/Gaussian parameters with TD detached;
            # a second path updates the TD branch with SD/Gaussian parameters detached.
            
            # 1. SD path: compute per-Gaussian spatially degraded colors using the
            # underwater image formation model.
            normalized_gs_distances = self._compute_normalized_gs_distances(means_crop, viewmat)

            spatial_degraded_colors_crop = (
                intrinsic_colors_crop * torch.exp(-attenuation_coeff * normalized_gs_distances)
                + ambient_light * (1 - torch.exp(-backscatter_coeff * normalized_gs_distances))
            )
            
            # 2. First spatiotemporal path: detach TD perturbations so gradients flow
            # only through SD/Gaussian parameters.
            spatiotemporal_degraded_colors_crop_for_sd_grad = (
                spatial_degraded_colors_crop + illumination_perturbation.detach().clone()
            )
            
            intrinsic_spatial_spatiotemporal_colors_crop = torch.cat(
                (
                    intrinsic_colors_crop,
                    spatial_degraded_colors_crop,
                    spatiotemporal_degraded_colors_crop_for_sd_grad,
                ),
                dim=-1,
            )

            # 3. First rasterization: render intrinsic, spatially degraded, and
            # spatiotemporal-degraded colors for the SD/Gaussian gradient path.
            intrinsic_spatial_spatiotemporal_render, alpha, self.info = rasterization(
                means=means_crop,
                quats=quats_crop,  # rasterization does normalization internally
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=intrinsic_spatial_spatiotemporal_colors_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
            if self.training:
                self.strategy.step_pre_backward(
                    self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
                )

            alpha = alpha[:, ...]
            background = self._get_background_color()
            
            rgb = intrinsic_spatial_spatiotemporal_render[:, ..., :3]
            rgb = torch.clamp(rgb, 0.0, 1.0)
            
            rgb_spatial_degraded = intrinsic_spatial_spatiotemporal_render[:, ..., 3:6]
            rgb_spatiotemporal_degraded_for_sd_grad = intrinsic_spatial_spatiotemporal_render[:, ..., 6:9]
            
            if self.config.enable_medium_bg:
                rgb_spatial_degraded = rgb_spatial_degraded + (1 - alpha) * ambient_light_image.unsqueeze(0)
                rgb_spatiotemporal_degraded_for_sd_grad = (
                    rgb_spatiotemporal_degraded_for_sd_grad
                    + (1 - alpha) * ambient_light_image.unsqueeze(0)
                )

            if render_mode == "RGB+ED":
                depth_im = intrinsic_spatial_spatiotemporal_render[:, ..., -1:]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
            else:
                depth_im = None            
            
            # 4. Second spatiotemporal path: detach SD/Gaussian quantities so gradients
            # flow only through the TD branch.
            spatiotemporal_degraded_colors_crop_for_td_grad = (
                spatial_degraded_colors_crop.detach().clone() + illumination_perturbation
            )
            
            safe_means = means_crop.detach().clone()
            safe_quats = quats_crop.detach().clone()
            safe_scales = torch.exp(scales_crop).detach().clone()
            safe_opacities = torch.sigmoid(opacities_crop).squeeze(-1).detach().clone()
            
            spatiotemporal_render_for_td_grad, _, _ = rasterization(
                means=safe_means,
                quats=safe_quats,
                scales=safe_scales,
                opacities=safe_opacities,
                colors=spatiotemporal_degraded_colors_crop_for_td_grad,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
            rgb_spatiotemporal_degraded_for_td_grad = spatiotemporal_render_for_td_grad[:, ..., :3]
            
            if self.config.enable_medium_bg:
                rgb_spatiotemporal_degraded_for_td_grad = (
                    rgb_spatiotemporal_degraded_for_td_grad
                    + (1 - alpha.detach()) * ambient_light_image.unsqueeze(0).detach()
                )
                
            return {
                "rgb": rgb.squeeze(0),  # type: ignore
                "rgb_spatial_degraded": rgb_spatial_degraded.squeeze(0),  # type: ignore
                "rgb_spatiotemporal_degraded_for_sd_grad": rgb_spatiotemporal_degraded_for_sd_grad.squeeze(0),  # type: ignore
                "rgb_spatiotemporal_degraded_for_td_grad": rgb_spatiotemporal_degraded_for_td_grad.squeeze(0),  # type: ignore

                "depth": depth_im,  # type: ignore
                "accumulation": alpha.squeeze(0),  # type: ignore
                "background": background,  # type: ignore
                
                "illumination_perturbation": illumination_perturbation,  # type: ignore
                
                "ambient_light_image": ambient_light_image,  # type: ignore
                "ambient_light": ambient_light,
                "attenuation_coeff": attenuation_coeff,
                "backscatter_coeff": backscatter_coeff,
            }  # type: ignore

        else:
            # Stage III: joint optimization of SD, TD, and Gaussian parameters.
            # 1. SD path:
            # Compute per-Gaussian spatially degraded colors using the underwater
            # image formation model.
            normalized_gs_distances = self._compute_normalized_gs_distances(means_crop, viewmat)

            spatial_degraded_colors_crop = (
                intrinsic_colors_crop * torch.exp(-attenuation_coeff * normalized_gs_distances)
                + ambient_light * (1 - torch.exp(-backscatter_coeff * normalized_gs_distances))
            )

            # 2. Add the TD branch perturbation to obtain spatiotemporal degraded colors.
            spatiotemporal_degraded_colors_crop = (
                spatial_degraded_colors_crop + illumination_perturbation
            )
            
            intrinsic_spatial_spatiotemporal_colors_crop = torch.cat(
                (intrinsic_colors_crop, spatial_degraded_colors_crop, spatiotemporal_degraded_colors_crop),
                dim=-1,
            )

            # 3. Single joint rasterization for intrinsic, spatially degraded, and
            # spatiotemporal-degraded colors.
            intrinsic_spatial_spatiotemporal_render, alpha, self.info = rasterization(
                means=means_crop,
                quats=quats_crop,  # rasterization does normalization internally
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=intrinsic_spatial_spatiotemporal_colors_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
            if self.training:
                self.strategy.step_pre_backward(
                    self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
                )

            alpha = alpha[:, ...]
            background = self._get_background_color()
            
            rgb = intrinsic_spatial_spatiotemporal_render[:, ..., :3]
            rgb = torch.clamp(rgb, 0.0, 1.0)
            
            rgb_spatial_degraded = intrinsic_spatial_spatiotemporal_render[:, ..., 3:6]
            rgb_spatiotemporal_degraded = intrinsic_spatial_spatiotemporal_render[:, ..., 6:9]

            if self.config.enable_medium_bg:
                rgb_spatial_degraded = rgb_spatial_degraded + (1 - alpha) * ambient_light_image.unsqueeze(0)
                rgb_spatiotemporal_degraded = (
                    rgb_spatiotemporal_degraded + (1 - alpha) * ambient_light_image.unsqueeze(0)
                )
                
            if render_mode == "RGB+ED":
                depth_im = intrinsic_spatial_spatiotemporal_render[:, ..., -1:]
                depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
            else:
                depth_im = None        

            return {
                "rgb": rgb.squeeze(0),  # type: ignore
                "rgb_spatial_degraded": rgb_spatial_degraded.squeeze(0),  # type: ignore
                "rgb_spatiotemporal_degraded": rgb_spatiotemporal_degraded.squeeze(0),  # type: ignore

                "depth": depth_im,  # type: ignore
                "accumulation": alpha.squeeze(0),  # type: ignore
                "background": background,  # type: ignore
                
                "illumination_perturbation": illumination_perturbation,  # type: ignore
                
                "ambient_light_image": ambient_light_image,  # type: ignore
                "ambient_light": ambient_light,
                "attenuation_coeff": attenuation_coeff,
                "backscatter_coeff": backscatter_coeff,
            }  # type: ignore
        
        
        
        # # ===============================================================================================================================================================================
    def get_gt_img(self, image: torch.Tensor, image_idx=None, use_cache=False):
        """
        Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
            image_idx: optional, index of the image for caching
            use_cache: optional, bool, whether to use caching mechanism
        """
        if not use_cache or image_idx is None:
            # If caching is disabled, or no image index is provided, process the image directly.
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            gt_img = self._downscale_if_required(image)
            return gt_img.to(self.device)

        # Use the cache when image_idx is available.
        downscale_factor = self._get_downscale_factor()
        cache_key = (image_idx, downscale_factor)

        if cache_key in self.gt_img_cache:
            # Return the cached result immediately.
            return self.gt_img_cache[cache_key].to(self.device)

        # Process the image once and store it in the cache.
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        self.gt_img_cache[cache_key] = gt_img   # Keep the cached tensor on its current device until retrieval.
        return gt_img

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"], image_idx=batch["image_idx"], use_cache=True), 
            outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        if self.config.enable_SD_branch:
            ambient_light = outputs["ambient_light"]
            attenuation_coeff = outputs["attenuation_coeff"]
            backscatter_coeff = outputs["backscatter_coeff"]
            
            for i in range(3):
                metrics_dict[f"ambient_light_{i}"] = ambient_light[0, i]
                metrics_dict[f"attenuation_coeff_{i}"] = attenuation_coeff[0, i]
                metrics_dict[f"backscatter_coeff_{i}"] = backscatter_coeff[0, i]               

        if self.config.enable_TD_branch:
            illumination_perturbation = outputs["illumination_perturbation"]
            mean_perturbation = illumination_perturbation.mean(dim=0)  # shape: [3]
            var_perturbation = illumination_perturbation.var(dim=0, unbiased=False)  # shape: [3]

            for i in range(3):
                metrics_dict[f"perturbation_mean_{i}"] = mean_perturbation[i]
                metrics_dict[f"perturbation_var_{i}"] = var_perturbation[i]                

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def _get_reconstruction_weight(self, pred_img: torch.Tensor) -> torch.Tensor:
        """Return the optional reconstruction weight map for an RGB prediction."""
        if self.config.enable_reg_loss and self.step > 1000:
            return 1 / (pred_img.detach() + 1e-3)
        return torch.ones_like(pred_img, device=pred_img.device)

    def _compute_reconstruction_loss(
        self,
        gt_img: torch.Tensor,
        pred_img: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the weighted L1 + SSIM reconstruction loss used by all stages."""
        if weight is None:
            weight = self._get_reconstruction_weight(pred_img)

        l1_loss = torch.abs((gt_img - pred_img) * weight).mean()

        gt_chw = gt_img.permute(2, 0, 1)[None, ...]
        pred_chw = pred_img.permute(2, 0, 1)[None, ...]
        weight_chw = weight.permute(2, 0, 1)[None, ...]

        if gt_chw.shape[-1] > 800 or gt_chw.shape[-2] > 800:
            ssim_loss = 1 - self.split_and_calculate_ssim(
                self.ssim,
                gt_chw * weight_chw,
                pred_chw * weight_chw,
            )
        else:
            ssim_loss = 1 - self.ssim(gt_chw * weight_chw, pred_chw * weight_chw)

        return (1 - self.config.ssim_lambda) * l1_loss + self.config.ssim_lambda * ssim_loss

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        rendered_depth = outputs["depth"]  # [H, W, 1]

        reset_every=self.config.reset_alpha_every * self.config.refine_every
        refine_stop_iter=self.config.stop_split_at
        depth_loss_enabled = self.config.enable_coarse_depth_supervision or self.config.enable_adaptive_depth_smoothness
        
        # ============================================================================== 

        if depth_loss_enabled:
            if "depth_image" not in batch:
                raise ValueError(
                    "MarineSTD-GS depth losses require batch['depth_image'], but it was not found."
                )
            if rendered_depth is None:
                raise ValueError(
                    "MarineSTD-GS depth losses require rendered depth. "
                    "Enable output_depth_during_training or keep depth losses enabled so get_outputs uses RGB+ED."
                )
            pseudo_depth = self.get_cached_pseudo_depth(batch["image_idx"], batch["depth_image"])
            pseudo_depth = pseudo_depth.to(self.device)

        # ============================================================================== 

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        mask = None
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
            # The current pipeline assumes depth_img has already been normalized before entering the model.

        # Multi-stage optimization losses. ==============================================================================
        if self.step < self.config.stage_1:
            # Stage I: supervise the SD path and the TD-only gradient path separately.
            pred_img_spatial_degraded = outputs["rgb_spatial_degraded"]
            pred_img_spatiotemporal_degraded_for_td_grad = outputs["rgb_spatiotemporal_degraded_for_td_grad"]
            if mask is not None:
                pred_img_spatial_degraded = pred_img_spatial_degraded * mask
                pred_img_spatiotemporal_degraded_for_td_grad = pred_img_spatiotemporal_degraded_for_td_grad * mask
                
            loss_dict = {
                "main_spatial_degraded": self._compute_reconstruction_loss(
                    gt_img, pred_img_spatial_degraded
                ),
                "main_spatiotemporal_degraded_for_td_grad": self._compute_reconstruction_loss(
                    gt_img, pred_img_spatiotemporal_degraded_for_td_grad
                ),
            }
        
        elif self.step >= self.config.stage_1 and self.step < self.config.stage_2:
            # Stage II: supervise both decoupled spatiotemporal gradient paths.
            pred_img_spatiotemporal_degraded_for_sd_grad = outputs["rgb_spatiotemporal_degraded_for_sd_grad"]
            pred_img_spatiotemporal_degraded_for_td_grad = outputs["rgb_spatiotemporal_degraded_for_td_grad"]
            if mask is not None:
                pred_img_spatiotemporal_degraded_for_sd_grad = pred_img_spatiotemporal_degraded_for_sd_grad * mask
                pred_img_spatiotemporal_degraded_for_td_grad = pred_img_spatiotemporal_degraded_for_td_grad * mask

            loss_dict = {
                "main_spatiotemporal_degraded_for_sd_grad": self._compute_reconstruction_loss(
                    gt_img, pred_img_spatiotemporal_degraded_for_sd_grad
                ),
                "main_spatiotemporal_degraded_for_td_grad": self._compute_reconstruction_loss(
                    gt_img, pred_img_spatiotemporal_degraded_for_td_grad
                ),
            }
        else:
            # Stage III: supervise the jointly optimized spatiotemporal image.
            pred_img_spatiotemporal_degraded = outputs["rgb_spatiotemporal_degraded"]
            if mask is not None:
                pred_img_spatiotemporal_degraded = pred_img_spatiotemporal_degraded * mask
        
            loss_dict = {
                "main_spatiotemporal_degraded": self._compute_reconstruction_loss(
                    gt_img, pred_img_spatiotemporal_degraded
                ),
            }

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict['scale_reg'] = scale_reg
        
        # Regularize the additive caustic illumination perturbation.
        perturbation_reg = torch.mean(outputs["illumination_perturbation"] ** 2)
        loss_dict['perturbation_reg'] = self.config.perturbation_regularization_weight * 1e-2 * perturbation_reg
        
        if self.config.enable_negative_perturbation_regularization:
            negative_perturbation_reg = negative_perturbation_regularization(
                perturbation=outputs["illumination_perturbation"],
                lambda_neg=1e-2,
                mode='relu_l2',
            )
            loss_dict['negative_perturbation_reg'] = (
                self.config.negative_perturbation_regularization_weight * negative_perturbation_reg
            )
        
        # Coarse Depth Supervision term L_cds.
        if self.config.enable_coarse_depth_supervision:
            pseudo_depth_flatten = pseudo_depth.flatten()               
            rendered_depth_flatten = rendered_depth.flatten()
            # The pseudo depth is disparity-like, so rendered depth is inverted before Pearson correlation.
            coarse_depth_supervision_loss = 1 - pearson_corrcoef(pseudo_depth_flatten, 1/(rendered_depth_flatten * 10  + 1))
            loss_dict['coarse_depth_supervision'] = coarse_depth_supervision_loss * self.config.weight_cds

        # Adaptive Edge-aware Depth Smoothness term L_ads.
        if self.config.enable_adaptive_depth_smoothness and  ( self.step % reset_every > 100 or self.step > refine_stop_iter): # type: ignore        
            adaptive_depth_smoothness_loss = adaptive_edge_aware_depth_smoothness_loss(
                depth_map=rendered_depth,                    # [H, W, 1]
                rgb_image=gt_img,                   # [H, W, 3]
                pseudo_depth=pseudo_depth,     # [H, W, 1]
                use_pseudo_for_mask=True,
                pseudo_already_normalized=True
            )
            loss_dict['adaptive_depth_smoothness'] = adaptive_depth_smoothness_loss * self.config.weight_ads
                    
        # ============================================================================== 


        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)                
        # Avoid calling camera.to(self.device) here. In this project setup, moving a camera
        # with image metadata can alter camera_to_worlds into an unexpected shape.
        outs = self.get_outputs(camera)

        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])      
        predicted_rgb = outputs["rgb"]
        if "rgb_spatiotemporal_degraded" in outputs:
            predicted_degraded_rgb = outputs["rgb_spatiotemporal_degraded"]
            degraded_metric_prefix = "spatiotemporal_degraded"
        elif "rgb_spatiotemporal_degraded_for_sd_grad" in outputs:
            predicted_degraded_rgb = outputs["rgb_spatiotemporal_degraded_for_sd_grad"]
            degraded_metric_prefix = "spatiotemporal_degraded_for_sd_grad"
        else:
            predicted_degraded_rgb = outputs["rgb_spatiotemporal_degraded_for_td_grad"]
            degraded_metric_prefix = "spatiotemporal_degraded_for_td_grad"
        cc_rgb = None

        # combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_rgb = torch.cat([gt_rgb, predicted_degraded_rgb, predicted_rgb], dim=1)
        images_dict = {"img": combined_rgb}

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        # ssim = self.ssim(gt_rgb, predicted_rgb)
        if gt_rgb.shape[-1] > 800 or predicted_rgb.shape[-2] > 800:
            ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, predicted_rgb)
        else:
            ssim = self.ssim(gt_rgb, predicted_rgb)          
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            if gt_rgb.shape[-1] > 800 or predicted_rgb.shape[-2] > 800:
                cc_ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, cc_rgb)
            else:
                cc_ssim = self.ssim(gt_rgb, cc_rgb)             
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        
        pred_degraded_img = predicted_degraded_rgb.permute(2, 0, 1)[None, ...]  
        degraded_psnr = self.psnr(gt_rgb, pred_degraded_img)
        if gt_rgb.shape[-1] > 800 or gt_rgb.shape[-2] > 800:
            degraded_ssim = self.split_and_calculate_ssim(self.ssim, gt_rgb, pred_degraded_img)
        else:
            degraded_ssim = self.ssim(gt_rgb, pred_degraded_img)  
        degraded_lpips = self.lpips(gt_rgb, pred_degraded_img.clamp(0.0, 1.0))   
        metrics_dict[f"{degraded_metric_prefix}_lpips"] = float(degraded_lpips)
        metrics_dict[f"{degraded_metric_prefix}_ssim"] = float(degraded_ssim)
        metrics_dict[f"{degraded_metric_prefix}_psnr"] = float(degraded_psnr.item())

           
        if outputs["ambient_light_image"] is not None:
            images_dict['ambient_light_image'] = outputs["ambient_light_image"]        

        if "rgb_attenuation_map" in outputs:
            images_dict["rgb_attenuation_map"] = outputs["rgb_attenuation_map"]
        if "rgb_backscatter_map" in outputs:
            images_dict["rgb_backscatter_map"] = outputs["rgb_backscatter_map"]
        if "rgb_caustic_pattern" in outputs:
            images_dict["rgb_caustic_pattern"] = outputs["rgb_caustic_pattern"]

        if "depth_image" in batch.keys():
            pseudo_depth = batch["depth_image"]  # [H,W,1] pseudo depth prior
            pseudo_depth = pseudo_depth / pseudo_depth.max() # Normalize to [0, 1].
            images_dict['pseudo_depth'] = pseudo_depth

        rendered_depth = outputs["depth"]
        q99 = torch.quantile(rendered_depth.flatten(), 0.99)  # Compute the 99th percentile for robust clipping.
        rendered_depth_clipped = torch.clamp(rendered_depth, max=q99)
        rendered_depth_normalized = rendered_depth_clipped / q99
        images_dict['rendered_depth_normalized'] = rendered_depth_normalized        
        
        return metrics_dict, images_dict


    def split_and_calculate_ssim(self, ssim_func, img1, img2):
        """Split img1 and img2 into four quadrants, compute SSIM per quadrant, and average the results."""
        # Get image height and width.
        H, W = img1.shape[-2], img1.shape[-1]

        # Split each image into four quadrants.
        h_half, w_half = H // 2, W // 2
        blocks1 = [
            img1[..., :h_half, :w_half],  # Top-left.
            img1[..., :h_half, w_half:],  # Top-right.
            img1[..., h_half:, :w_half],  # Bottom-left.
            img1[..., h_half:, w_half:]   # Bottom-right.
        ]
        blocks2 = [
            img2[..., :h_half, :w_half],  # Matching quadrants from the second image.
            img2[..., :h_half, w_half:],
            img2[..., h_half:, :w_half],
            img2[..., h_half:, w_half:]
        ]

        # Compute SSIM for each quadrant separately.
        ssim_values = [ssim_func(b1, b2) for b1, b2 in zip(blocks1, blocks2)]

        # Return the mean SSIM across all quadrants.
        return torch.mean(torch.stack(ssim_values))
    
    def get_cached_pseudo_depth(self, image_idx, depth_image):
        downscale_factor = self._get_downscale_factor()
        cache_key = (image_idx, downscale_factor)

        if cache_key in self.depth_cache:
            # Return the cached pseudo depth immediately.
            return self.depth_cache[cache_key]
        else:
            # Resize once and cache the result for reuse.
            pseudo_depth_resized = self._downscale_if_required(depth_image)
            pseudo_depth_resized = pseudo_depth_resized / pseudo_depth_resized.max()   # Normalize to [0, 1].

            # Cache the resized pseudo depth.
            self.depth_cache[cache_key] = pseudo_depth_resized
            # print(f'cache {cache_key}')
            return pseudo_depth_resized
    
    
