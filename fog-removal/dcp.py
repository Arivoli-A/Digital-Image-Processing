import cv2
import math
import numpy as np


class DarkChannelPrior:
    def __init__(
        self,
        patch_size=15,
        omega=0.95,
        guided_filter_radius=60,
        guided_filter_eps=0.0001,
        t_min=0.1,
        atm_percentile=0.001,
    ):
        self.patch_size = patch_size
        self.omega = omega
        self.guided_filter_radius = guided_filter_radius
        self.guided_filter_eps = guided_filter_eps
        self.t_min = t_min
        self.atm_percentile = atm_percentile

    def dehaze(self, image):
        # 0) Normalize image to [0, 1]
        normalized_img = image.astype(np.float64) / 255.0

        # 1) Find dark channel
        dark = self._compute_dark_channel(normalized_img)

        # 2) Find A (atmospheric light)
        A = self._compute_atmospheric_light(normalized_img, dark)

        # 3) Estimate transmission
        t_es = self._compute_transmission(normalized_img, A)

        # 4) Refine transmisison (guided filtering)
        t = self._refine_transmission(image, t_es)

        # 5) Recover dehazed image
        recovered = self._recover_scene(normalized_img, t, A)

        # 6) Convert recovered image back to int8
        recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)

        return recovered

    def _compute_dark_channel(self, image):
        b, g, r = cv2.split(image)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark = cv2.erode(dc, kernel)

        return dark

    def _compute_atmospheric_light(self, image, dark):
        [h, w] = image.shape[:2]
        img_size = h * w
        num_pixels = int(max(math.floor(img_size * self.atm_percentile), 1))
        dark_vec = dark.reshape(img_size)
        image_vec = image.reshape(img_size, 3)

        # Get indices of brightest pixels in dark channel
        indices = dark_vec.argsort()[-num_pixels:]

        # Average the corresponding RGB values
        atm_sum = np.zeros([1, 3])
        for idx in indices:
            atm_sum += image_vec[idx]

        A = atm_sum / num_pixels
        return A

    def _compute_transmission(self, image, A):
        # Normalize by atmospheric light
        img_normalized = np.empty_like(image)
        for c in range(3):
            img_normalized[:, :, c] = image[:, :, c] / A[0, c]

        # Compute dark channel of normalized image
        dark = self._compute_dark_channel(img_normalized)

        # Estimate transmission
        t = 1 - self.omega * dark
        return t

    def _refine_transmission(self, image, t):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255.0

        refined = self._guided_filter(
            gray, t, self.guided_filter_radius, self.guided_filter_eps
        )
        return refined

    def _guided_filter(self, guide, src, radius, eps):
        mean_I = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

        q = mean_a * guide + mean_b
        return q

    def _recover_scene(self, image, t, A):
        result = np.empty_like(image)
        t = cv2.max(t, self.t_min)

        for c in range(3):
            result[:, :, c] = (image[:, :, c] - A[0, c]) / t + A[0, c]

        return result


if __name__ == "__main__":
    dehazer = DarkChannelPrior()
    foggy_img = cv2.imread("fog_test_2.png")
    dehazed_img = dehazer.dehaze(foggy_img)
    cv2.imwrite("dehazed.jpg", dehazed_img)
