import pywt
import cv2
import numpy as np

from dcp import DarkChannelPrior


class KimDefogPipeline:
    def __init__(
        self,
        fusion_weight=0.5,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(10, 10),
        wavelet="db4",
        dwt_level=1,
        sharpening_factor=0.7,
        **dcp_kwargs
    ):
        """
        Args:
            fusion_weight (float): Weight for fusion (0.0 to 1.0).
            clahe_clip_limit (float): Threshold for CLAHE contrast limiting.
            clahe_tile_grid_size (tuple): Grid size for CLAHE.
            wavelet (str): Wavelet family (e.g., 'db4', 'haar').
            dwt_level (int): Number of decomposition levels for DWT.
            sharpening_factor (float): Scaling factor for sharpening.
            **dcp_kwargs: Arbitrary arguments passed to DarkChannelPrior
                          (e.g., radius, omega, etc.)
        """
        self.fusion_weight = fusion_weight
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.wavelet = wavelet
        self.dwt_level = dwt_level
        self.sharpening_factor = sharpening_factor

        # Pass any extra arguments (like radius or method) to the DCP class
        self.dehazer = DarkChannelPrior(**dcp_kwargs)

    def dehaze(self, image, return_baseline=False):
        # 1) Apply DCP (Baseline)
        dehazed_img_baseline = self.dehazer.dehaze(image)

        # If we only want the baseline for ablation study, return it here
        if return_baseline:
            return dehazed_img_baseline

        # 2) Convert to HSV
        hsv_img = cv2.cvtColor(dehazed_img_baseline, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        original_shape = v.shape

        # 3a) Apply CLAHE
        v_clahe = self.clahe(v)

        # 3b) Apply Multi-Level DWT
        v_dwt = self.dwt_multilevel(v)

        # Ensure shapes match after reconstruction
        v_dwt = v_dwt[: original_shape[0], : original_shape[1]]

        # 4) Fusion
        v_fused = self.fusion_weight * v_clahe + (1 - self.fusion_weight) * v_dwt
        v_fused = np.clip(v_fused, 0, 255).astype(np.uint8)

        # 5) Reconstruct
        fused_image = cv2.merge([h, s, v_fused])
        fused_image = cv2.cvtColor(fused_image, cv2.COLOR_HSV2BGR)

        return fused_image

    def clahe(self, v):
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size
        )
        return clahe.apply(v)

    def dwt_multilevel(self, v):
        coeffs = pywt.wavedec2(v, self.wavelet, level=self.dwt_level)
        cA = coeffs[0]
        details = coeffs[1:]

        cA_sharpened = self.laplacian_sharpening(cA)

        # Precompute threshold constant factor
        # sqrt(2 * log(N)) only depends on image size, not the specific wavelet band
        visu_shrink_factor = np.sqrt(2 * np.log(v.size))

        new_details = []
        for cH, cV, cD in details:
            sigma = np.median(np.abs(cD)) / 0.6745

            # Use precomputed factor
            threshold = sigma * visu_shrink_factor

            # pywt.threshold is already efficient C-code
            cH_den = pywt.threshold(cH, threshold, mode="soft")
            cV_den = pywt.threshold(cV, threshold, mode="soft")
            cD_den = pywt.threshold(cD, threshold, mode="soft")

            new_details.append((cH_den, cV_den, cD_den))

        new_coeffs = [cA_sharpened] + new_details
        v_rec = pywt.waverec2(new_coeffs, self.wavelet)

        return np.clip(v_rec, 0, 255).astype(np.uint8)

    def laplacian_sharpening(self, cA):
        cA_uint8 = np.clip(cA, 0, 255).astype(np.uint8)
        # Handle small cA sizes at deep levels
        ksize = 3
        if cA.shape[0] < 3 or cA.shape[1] < 3:
            # If decomposition is too deep, skip sharpening or use kernel size 1
            return cA

        laplacian = cv2.Laplacian(cA_uint8, cv2.CV_64F, ksize=ksize)
        cA_sharpened = cA.astype(np.float64) - self.sharpening_factor * laplacian
        return cA_sharpened


if __name__ == "__main__":
    # --- ABLATION STUDY EXAMPLES ---

    # 1. Baseline vs Hybrid
    pipeline = KimDefogPipeline()
    img = cv2.imread("fog_test_2.png")

    baseline = pipeline.dehaze(img, return_baseline=True)  # Just DCP
    hybrid = pipeline.dehaze(img, return_baseline=False)  # Full Pipeline

    # 2. Varying DWT Family and Levels
    pipeline_haar_lvl3 = KimDefogPipeline(wavelet="haar", dwt_level=3)

    # 3. Varying DCP Internals (Atmospheric light, radius)
    # This presumes your DarkChannelPrior __init__ accepts these args
    pipeline_custom_dcp = KimDefogPipeline(
        radius=30, omega=0.85  # passed to DCP  # passed to DCP
    )
