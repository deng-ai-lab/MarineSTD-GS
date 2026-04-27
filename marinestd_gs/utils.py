
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor



class InstantaneousBrightnessFeatureEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super().__init__()

        # x0: full-resolution features
        self.conv0 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # x1: H/2
        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # x2: H/4
        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # x3: H/8
        self.down3 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Global feature encoding
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, image, device):
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

        # x0: full-resolution features
        x0 = self.conv0(image)  # [1, C, H, W]

        # x1: H/2
        x1_down = self.down1(x0)
        x1 = F.relu(self.conv1(x1_down) + x1_down)

        # x2: H/4
        x2_down = self.down2(x1)
        x2 = F.relu(self.conv2(x2_down) + x2_down)

        # x3: H/8
        x3_down = self.down3(x2)
        x3 = F.relu(self.conv3(x3_down) + x3_down)

        # Global encoding
        global_feat = self.global_fc(x3)

        return {
            'global_feature': global_feat.squeeze(0),  # f_g in the paper, [C]
            'local_feature_map': x3.squeeze(0),        # F_l in the paper, [C, H/8, W/8]
        }



class WaterParametersExtractor(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=32, output_dim=9):
        super().__init__()
        input_channels = input_channels * 2


        def dw_pw(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # DW
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),  # PW
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            dw_pw(input_channels, hidden_dim),
            nn.AvgPool2d(2),

            dw_pw(hidden_dim, hidden_dim * 2),
            nn.AvgPool2d(2),

            dw_pw(hidden_dim * 2, hidden_dim * 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, degraded_image, pseudo_depth, device):
        degraded_image = degraded_image.to(device)
        pseudo_depth = pseudo_depth.to(device)

        pseudo_depth = (pseudo_depth - pseudo_depth.min()) / (pseudo_depth.max() - pseudo_depth.min() + 1e-8)
        depth_aware_enhanced_image = degraded_image * (1 - pseudo_depth)
        wpe_input = torch.cat([degraded_image, depth_aware_enhanced_image], dim=-1)  # [H, W, 6]

        wpe_input = wpe_input.permute(2, 0, 1).unsqueeze(0)  # [1, 6, H, W]

        # Extract global features
        x = self.encoder(wpe_input)
        global_feat = self.fc(x)

        return global_feat




def project_and_sample_local_feature(
    points_world: torch.Tensor,        # [N, 3]
    viewmat: torch.Tensor,             # [4, 4]
    K: torch.Tensor,                   # [3, 3]
    feature_map: torch.Tensor,         # feature map [C, H_i, W_i]
    image_resolution: tuple           # (H, W)
):
    """
    Project world points to the image plane and sample the local feature map.

    Returns:
    - sampled_feature: [N, C], with invalid projected points filled by zero
    - valid_mask: [N] bool tensor indicating valid projected points
    """
    device = points_world.device
    N = points_world.shape[0]
    H, W = image_resolution

    assert len(viewmat.shape) == 2 and len(K.shape) == 2, f'viewmat.shape is {viewmat.shape} and K.shape is {K.shape}'
    try:
        # === Step 1. World coordinates -> camera coordinates ===
        points_h = torch.cat([points_world, torch.ones((N, 1), device=device)], dim=-1)  # [N, 4]
        points_cam = (points_h @ viewmat.T)[:, :3]                                       # [N, 3]

        # === Step 2. Camera coordinates -> image-plane pixel coordinates ===
        uvz = points_cam @ K.T  # [N, 3]
        x = uvz[:, 0] / (uvz[:, 2] + 1e-6)
        y = uvz[:, 1] / (uvz[:, 2] + 1e-6)


        if torch.isnan(uvz).any() or torch.isinf(uvz).any():
            
            print("points_world stats:")
            print(" - shape:", points_world.shape)
            print(" - NaN:", torch.isnan(points_world).any().item())
            print(" - Inf:", torch.isinf(points_world).any().item())
            print(" - min:", points_world.min().item(), "max:", points_world.max().item())
            print(" - sample:", points_world[torch.randperm(points_world.shape[0])[:5]])
            
            print("WARNING: uvz contains NaN or Inf!")
            print("uvz stats: min =", uvz.min().item(), "max =", uvz.max().item())
            print("points_cam stats:", points_cam.min().item(), points_cam.max().item())
            print("K:", K)
            print("viewmat:", viewmat)
            raise ValueError("Invalid values detected in uvz computation.")
            
        # === Step 3. Filter visible points ===
        mask_x = (x >= 0) & (x < W)
        mask_y = (y >= 0) & (y < H)
        mask_z = uvz[:, 2] > 0
        valid_mask = mask_x & mask_y & mask_z

        if valid_mask.sum() == 0:
            raise ValueError("No valid projected points within image bounds and in front of the camera.")

        # === Step 4. Normalize projected coordinates to the original image range ===
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        x_norm = (x_valid / (W - 1)) * 2 - 1
        y_norm = (y_valid / (H - 1)) * 2 - 1
        normalized_coords = torch.stack([x_norm, y_norm], dim=-1)  # [N_valid, 2]
        normalized_coords = normalized_coords[None, None, :, :]    # [1, 1, N_valid, 2]

        # === Step 5. Sample the local feature map ===
        try:
            feature_map = feature_map.unsqueeze(0)  # [1, C, H_i, W_i]
            sampled = F.grid_sample(
                feature_map, normalized_coords, mode='bilinear',
                padding_mode='border', align_corners=True
            )  # -> [1, C, 1, N_valid]
            sampled = sampled.squeeze(2).squeeze(0).T  # [N_valid, C]

            sampled_feature = torch.zeros((N, sampled.shape[1]), device=device)
            sampled_feature[valid_mask] = sampled

        except Exception as e:
            print("Error during local feature map sampling:", e)
            print(f"feature_map shape: {feature_map.shape}")
            print(f"normalized_coords shape: {normalized_coords.shape}")
            if 'sampled' in locals():
                print(f"sampled shape: {sampled.shape}")
            raise

        return sampled_feature, valid_mask

    except Exception as e:
        print("==== Error in project_and_sample_local_feature ====")
        print("points_world shape:", points_world.shape)
        print("viewmat shape:", viewmat.shape)
        print("K shape:", K.shape)
        print("image_resolution:", image_resolution)
        if 'x' in locals():
            print("x range:", x.min().item(), x.max().item())
        if 'y' in locals():
            print("y range:", y.min().item(), y.max().item())
        if 'valid_mask' in locals():
            print("valid points:", valid_mask.sum().item())
        raise
    
    
def negative_perturbation_regularization(perturbation, lambda_neg=1e-2, mode='relu_l2'):
    """
    Regularization for negative additive illumination perturbation values.

    Args:
        perturbation (Tensor): Additive illumination perturbation tensor.
        lambda_neg (float): Weight for the negative regularization term.
        mode (str): Regularization mode, one of ['relu_l2', 'relu_l1'].
                    - 'relu_l2': penalize squared negative values (default)
                    - 'relu_l1': penalize absolute negative values

    Returns:
        Tensor: Negative perturbation regularization loss.
    """
    # Only select negative values (ReLU)
    neg_perturbation = torch.clamp(-perturbation, min=0.0)

    if mode == 'relu_l2':
        neg_loss = torch.mean(neg_perturbation ** 2)
    elif mode == 'relu_l1':
        neg_loss = torch.mean(neg_perturbation)
    else:
        raise ValueError(f"Unsupported mode for negative perturbation regularization: {mode}")
    return lambda_neg * neg_loss


def gradient_x(img):
    return img[..., :-1, :] - img[..., 1:, :]

def gradient_y(img):
    return img[..., :, :-1] - img[..., :, 1:]




def to_grayscale(img):
    """Assumes input shape [1, C, H, W]"""
    if img.shape[1] == 3:
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        return img

def normalize_depth(depth):
    """Min-max normalize a depth map to [0, 1]"""
    d_min = depth.amin(dim=[2, 3], keepdim=True)
    d_max = depth.amax(dim=[2, 3], keepdim=True)
    return (depth - d_min) / (d_max - d_min + 1e-8)


def adaptive_edge_aware_depth_smoothness_loss(
    depth_map: torch.Tensor,
    rgb_image: torch.Tensor,
    pseudo_depth: torch.Tensor,
    use_pseudo_for_mask: bool = True,
    pseudo_already_normalized: bool = True,
):
    """
    Args:
        depth_map:      [H, W, 1] predicted depth from the model
        rgb_image:      [H, W, 3] input RGB image (visible light)
        pseudo_depth:   [H, W, 1] auxiliary pseudo depth map (disparity-like in this codebase)
        use_pseudo_for_mask: whether to use pseudo_depth to guide far_mask
        pseudo_already_normalized: whether pseudo_depth ∈ [0,1]
    Returns:
        Scalar smoothness loss
    """

    # Reshape to [1, C, H, W]
    depth_map = depth_map.permute(2, 0, 1).unsqueeze(0)        # [1, 1, H, W]
    rgb_image = rgb_image.permute(2, 0, 1).unsqueeze(0)        # [1, 3, H, W]
    pseudo_depth = pseudo_depth.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

    # Convert to grayscale
    rgb_gray = to_grayscale(rgb_image)
    pseudo_gray = to_grayscale(pseudo_depth)

    # Compute gradients
    depth_grad_x = gradient_x(depth_map)
    depth_grad_y = gradient_y(depth_map)

    rgb_grad_x = gradient_x(rgb_gray)
    rgb_grad_y = gradient_y(rgb_gray)

    pseudo_grad_x = gradient_x(pseudo_gray)
    pseudo_grad_y = gradient_y(pseudo_gray)

    # Edge-aware weights
    weight_rgb_x = torch.exp(-torch.abs(rgb_grad_x))
    weight_rgb_y = torch.exp(-torch.abs(rgb_grad_y))

    weight_depth_x = torch.exp(-torch.abs(pseudo_grad_x))
    weight_depth_y = torch.exp(-torch.abs(pseudo_grad_y))

    # Compute far_mask. The stored pseudo depth is disparity-like (near large, far small),
    # so 1 - pseudo_depth matches the paper's far-large depth convention.
    with torch.no_grad():
        if use_pseudo_for_mask:
            if pseudo_already_normalized:
                far_mask = (1.0 - pseudo_depth).detach()  # [1, 1, H, W]
            else:
                norm = normalize_depth(pseudo_depth)
                far_mask = (1.0 - norm).detach()
        else:
            far_mask = normalize_depth(depth_map).detach()  # [1, 1, H, W]

    # Crop far_mask to match gradients
    far_mask_x = far_mask[:, :, :weight_depth_x.shape[2], :weight_depth_x.shape[3]]
    far_mask_y = far_mask[:, :, :weight_depth_y.shape[2], :weight_depth_y.shape[3]]

    # Weight fusion: far uses more RGB guidance, near uses more depth guidance
    final_weight_x = (1 - far_mask_x) * weight_depth_x + far_mask_x * weight_rgb_x
    final_weight_y = (1 - far_mask_y) * weight_depth_y + far_mask_y * weight_rgb_y

    # Apply to depth gradients
    smoothness_x = depth_grad_x * final_weight_x
    smoothness_y = depth_grad_y * final_weight_y

    # Final loss
    loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
    return loss
