import pywt
import cv2
import numpy as np

from dcp import DarkChannelPrior


class KimDefogPipeline:
    def __init__(self, fusion_weight=0.5):
        self.fusion_weight = fusion_weight
        self.dehazer = DarkChannelPrior()

    def dehaze(self, image):
        # 1) Apply DCP with Guided Filtering to get an initial dehazed image
        dehazed_img = self.dehazer.dehaze(image)

        # 2) Covert image from BGR to HSV
        hsv_img = cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        original_shape = v.shape

        # 3a) Apply CLAHE on the intensity (v) channel of the dehazed image to increase contrast
        v_clahe = self.clahe(v)

        # 3b) Apply DWT on the intensity (v) channel of the dehazed image
        v_dwt = self.dwt(v)
        v_dwt = v_dwt[: original_shape[0], : original_shape[1]]

        # 4) Merge the V channels from CLAHE and DWT back together
        v_fused = self.fusion_weight * v_clahe + (1 - self.fusion_weight) * v_dwt
        v_fused = np.clip(v_fused, 0, 255).astype(np.uint8)

        # 5) Create the final image
        fused_image = cv2.merge([h, s, v_fused])
        fused_image = cv2.cvtColor(fused_image, cv2.COLOR_HSV2BGR)

        return fused_image

    def clahe(self, v):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        v_clahe = clahe.apply(v)

        return v_clahe

    def dwt(self, v):
        # DWT with the Daubechies transform
        v_float = v.astype(np.float64)
        coeffs = pywt.dwt2(v, "db4")
        cA, (cH, cV, cD) = coeffs

        # Apply sharpening to the low frequency coefficient (cA)
        cA_sharpened = self.laplacian_sharpening(cA)

        # Apply denoising to the high frequency coefficients (cH, cV, cD)
        sigma = np.median(np.abs(cD)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(v.size))
        cH_denoised, cV_denoised, cD_denoised = self.soft_thresholding(
            cH, cV, cD, threshold
        )

        # Inverse DWT
        new_coeffs = (cA_sharpened, (cH_denoised, cV_denoised, cD_denoised))
        v_dwt = pywt.idwt2(new_coeffs, "db4")
        v_dwt = np.clip(v_dwt, 0, 255).astype(np.uint8)

        return v_dwt

    def laplacian_sharpening(self, cA):
        cA_uint8 = np.clip(cA, 0, 255).astype(np.uint8)
        laplacian = cv2.Laplacian(cA_uint8, cv2.CV_64F)
        cA_sharpened = cA.astype(np.float64) - 0.7 * laplacian
        cA_sharpened = np.clip(cA_sharpened, 0, 255)

        return cA_sharpened

    def soft_thresholding(self, cH, cV, cD, threshold):
        cH_denoised = pywt.threshold(cH, threshold, mode="soft")
        cV_denoised = pywt.threshold(cV, threshold, mode="soft")
        cD_denoised = pywt.threshold(cD, threshold, mode="soft")

        return cH_denoised, cV_denoised, cD_denoised


if __name__ == "__main__":
    dehazer = KimDefogPipeline()
    foggy_img = cv2.imread("fog_test_2.png")
    dehazed_img = dehazer.dehaze(foggy_img)
    cv2.imwrite("kim_dehazed.jpg", dehazed_img)
