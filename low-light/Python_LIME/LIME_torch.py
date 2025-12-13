import cv2
import torch
from . import LIME_functions as LIME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lime(image, weight_strategy=3, gamma=0.4, std_dev=0.04):
    """
    Process a single image using the LIME enhancement pipeline.

    Parameters
    ----------
    image : np.ndarray
        Image in BGR format (H, W, 3), dtype uint8 or float32.
    weight_strategy : int
        Weight strategy for illumination map update.
    gamma : float
        Gamma for correction.
    std_dev : float
        Standard deviation parameter for BM3D denoising.

    Returns
    -------
    denoised_image : np.ndarray
        Enhanced and denoised image in RGB (float32, range 0–1).
    """

    # Convert to RGB and move to GPU for enhancement
    image_rgb = torch.tensor(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0,
        dtype=torch.float32
    ).permute(2, 0, 1).to(device)  # CHW

    # Compute illumination map on GPU
    illumination_map = torch.max(image_rgb, dim=0).values

    # --- Solve illumination map on CPU to avoid GPU OOM ---
    updated_ill_map_cpu = LIME.update_illumination_map(
        illumination_map.cpu().numpy(), weight_strategy
    )
    corrected_ill_map = LIME.gamma_correction(
        torch.tensor(updated_ill_map_cpu, dtype=torch.float32, device=device).abs(), gamma
    ).unsqueeze(0)

    # Image enhancement on GPU
    new_image = image_rgb / corrected_ill_map
    new_image = torch.clamp(new_image, 0, 1)

    # BM3D denoising (CPU only)
    new_image_cpu = new_image.permute(1, 2, 0).cpu().numpy()  # HWC
    corrected_ill_map_cpu = corrected_ill_map.squeeze(0).cpu().numpy()  # HW
    denoised_image_cpu = LIME.bm3d_yuv_denoising(new_image_cpu, corrected_ill_map_cpu, std_dev)

    # Convert back to tensor if needed or just return NumPy
    denoised_image = torch.tensor(
        denoised_image_cpu, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    return denoised_image.permute(1, 2, 0).cpu().numpy()  # HWC
