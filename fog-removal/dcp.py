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
        # 0) Normalize to [0, 1] - Math operations are faster on float32 than float64
        # and usually sufficient for image processing.
        normalized_img = image.astype(np.float32) / 255.0

        # 1) Find dark channel
        dark = self._compute_dark_channel(normalized_img)

        # 2) Find A (atmospheric light)
        A = self._compute_atmospheric_light(normalized_img, dark)

        # 3) Estimate transmission
        t_es = self._compute_transmission(normalized_img, A)

        # 4) Refine transmission (guided filtering)
        t = self._refine_transmission(image, t_es)

        # 5) Recover dehazed image
        recovered = self._recover_scene(normalized_img, t, A)

        # 6) Clip and convert back to uint8
        return np.clip(recovered * 255, 0, 255).astype(np.uint8)

    def _compute_dark_channel(self, image):
        # Optimization: np.min over axis 2 avoids creating r,g,b splits
        dc = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        return cv2.erode(dc, kernel)

    def _compute_atmospheric_light(self, image, dark):
        h, w = image.shape[:2]
        img_size = h * w
        num_pixels = int(max(math.floor(img_size * self.atm_percentile), 1))

        # Optimization: Flatten arrays
        dark_vec = dark.ravel()
        image_vec = image.reshape(img_size, 3)

        # Optimization: Use argpartition (O(N)) instead of argsort (O(N log N))
        # We only need the top K largest values, we don't care about their internal order.
        indices = np.argpartition(dark_vec, -num_pixels)[-num_pixels:]

        # Vectorized mean
        A = np.mean(image_vec[indices], axis=0)
        return A.reshape(1, 3)

    def _compute_transmission(self, image, A):
        # Optimization: Vectorized normalization
        # image is (H,W,3), A is (1,3). Broadcasting handles the division.
        img_normalized = image / A

        dark = self._compute_dark_channel(img_normalized)
        return 1 - self.omega * dark

    def _refine_transmission(self, image, t):
        # Optimization: Convert to float32 gray once
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Guided filter (if you have opencv-contrib, use cv2.ximgproc.guidedFilter for max speed)
        # implementation below is the vectorized numpy version
        return self._guided_filter(
            gray, t, self.guided_filter_radius, self.guided_filter_eps
        )

    def _guided_filter(self, guide, src, radius, eps):
        # Box filter is fast (O(1) with integral images), but ensuring types match is key
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

        return mean_a * guide + mean_b

    def _recover_scene(self, image, t, A):
        # Optimization: Vectorize the recovery formula
        # t is (H,W), needs to be (H,W,1) to broadcast against (H,W,3)
        t_bound = cv2.max(t, self.t_min)
        t_expanded = t_bound[:, :, np.newaxis]

        # Result = (I - A) / t + A
        return (image - A) / t_expanded + A
