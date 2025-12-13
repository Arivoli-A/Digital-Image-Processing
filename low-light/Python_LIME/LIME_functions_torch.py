import cv2
import numpy as np

from os import listdir
from typing import Union

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from bm3d import bm3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 1. Sparse forward-difference matrices (d_x, d_y)

def d_sparse_matrices(illumination_map: torch.Tensor):
    M, N = illumination_map.shape
    size = M * N

    # Horizontal difference
    rows, cols, vals = [], [], []
    for i in range(size):
        rows.append(i); cols.append(i); vals.append(-1.0)
        if (i + 1) % N != 0:
            rows.append(i); cols.append(i + 1); vals.append(1.0)

    idx_x = torch.tensor([rows, cols], dtype=torch.long, device=device)
    val_x = torch.tensor(vals, dtype=torch.float32, device=device)
    d_x = torch.sparse_coo_tensor(idx_x, val_x, (size, size), device=device).coalesce()

    # Vertical difference
    rows, cols, vals = [], [], []
    for i in range(size):
        rows.append(i); cols.append(i); vals.append(-1.0)
        if i + N < size:
            rows.append(i); cols.append(i + N); vals.append(1.0)

    idx_y = torch.tensor([rows, cols], dtype=torch.long, device=device)
    val_y = torch.tensor(vals, dtype=torch.float32, device=device)
    d_y = torch.sparse_coo_tensor(idx_y, val_y, (size, size), device=device).coalesce()

    return d_x, d_y


# 2. Partial derivatives

def partial_derivative_vectorized(input_matrix: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == 'x':
        grad = torch.roll(input_matrix, shifts=-1, dims=1) - input_matrix
        grad[:, -1] = 0
    elif direction == 'y':
        grad = torch.roll(input_matrix, shifts=-1, dims=0) - input_matrix
        grad[-1, :] = 0
    else:
        raise ValueError("Direction must be 'x' or 'y'.")
    return grad


# 3. Gaussian weight

def gaussian_blur(image: torch.Tensor, sigma: float, ksize: int):
    radius = ksize // 2
    x = torch.arange(-radius, radius + 1, device=device)
    g = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()

    kernel = kernel.unsqueeze(0).unsqueeze(0)
    img = image.unsqueeze(0).unsqueeze(0)
    return F.conv2d(img, kernel, padding=radius).squeeze()


def gaussian_weight(grad: torch.Tensor, size: int, sigma: float, epsilon: float):
    denom = epsilon + gaussian_blur(torch.abs(grad), sigma, size)
    weights = gaussian_blur(1.0 / denom, sigma, size)
    return weights



# 4. Initialize weights

def initialize_weights(ill_map: torch.Tensor, strategy_n: int, epsilon: float = 0.001):
    if strategy_n == 1:
        weights_x = torch.ones_like(ill_map)
        weights_y = torch.ones_like(ill_map)
    else:
        grad_t_x = partial_derivative_vectorized(ill_map, 'x')
        grad_t_y = partial_derivative_vectorized(ill_map, 'y')

        if strategy_n == 2:
            weights_x = 1 / (torch.abs(grad_t_x) + epsilon)
            weights_y = 1 / (torch.abs(grad_t_y) + epsilon)
        else:
            sigma, size = 2, 15
            weights_x = gaussian_weight(grad_t_x, size, sigma, epsilon)
            weights_y = gaussian_weight(grad_t_y, size, sigma, epsilon)

    grad_t_x = partial_derivative_vectorized(ill_map, 'x')
    grad_t_y = partial_derivative_vectorized(ill_map, 'y')

    modified_w_x = weights_x / (torch.abs(grad_t_x) + epsilon)
    modified_w_y = weights_y / (torch.abs(grad_t_y) + epsilon)

    return modified_w_x.flatten(), modified_w_y.flatten()


# 5. Solve illumination map update

def update_illumination_map(ill_map: torch.Tensor, weight_strategy: int = 3):
    t_vec = ill_map.flatten().unsqueeze(1)

    d_x, d_y = d_sparse_matrices(ill_map)
    w_x, w_y = initialize_weights(ill_map, weight_strategy)

    W_x = torch.diag(w_x).to(device)
    W_y = torch.diag(w_y).to(device)

    X = d_x.transpose(0, 1).to_dense() @ W_x @ d_x.to_dense()
    Y = d_y.transpose(0, 1).to_dense() @ W_y @ d_y.to_dense()

    alpha = 0.15
    I = torch.eye(X.shape[0], device=device)

    A = I + alpha * (X + Y)

    sol = torch.linalg.solve(A, t_vec)
    return sol.reshape(ill_map.shape)

# 6. Gamma correction

def gamma_correction(ill_map: torch.Tensor, gamma: float):
    return ill_map ** gamma


# 7. BM3D YUV denoising 

def bm3d_yuv_denoising(image: torch.Tensor, cor_ill_map: torch.Tensor, std_dev=0.02):
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # YUV conversion
    image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0]

    den_y = bm3d(y_channel, std_dev)

    image_yuv[:, :, 0] = den_y
    den_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    den_rgb_torch = torch.tensor(den_rgb, dtype=torch.float32, device=device) / 255.0

    out = image * cor_ill_map + den_rgb_torch.permute(2, 0, 1) * (1 - cor_ill_map)
    return out.clamp(0, 1)


# 8. Lightness Order Error (LOE)

def loss_calculation(reference_image: torch.Tensor, refined_image: torch.Tensor) -> float:
    ref_flat = reference_image.flatten()
    refd_flat = refined_image.flatten()

    ref_order = ref_flat[:, None] >= ref_flat[None, :]
    refd_order = refd_flat[:, None] >= refd_flat[None, :]

    xor_term = torch.logical_xor(ref_order, refd_order)
    loss = xor_term.float().sum().item()

    N = ref_flat.numel()
    return loss / (N * N * 1000)

