import cv2
import numpy as np


class TestMatcher:
    def __init__(self, config):
        self.img = cv2.imread(config["img_path"])
        self.keys, self.descs = self._apply_sift(self.img)
        c = config["c_crop"]
        self.c_crop = (slice(c[1], c[3]), slice(c[0], c[2]))
        t = config["t_crop"]
        self.t_crop = (slice(t[1], t[3]), slice(t[0], t[2]))

    def _apply_sift(self, img):
        sift = cv2.xfeatures2d.SIFT_create()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts, descs = sift.detectAndCompute(img, None)
        return pts, descs

    def _match_descs(self, descs, min_matches=50, max_matches=150, ratio=0.75):
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(self.descs, descs, k=2)
        matches = [m for m, n in matches if m.distance < ratio * n.distance]
        print("Number of Matches:", len(matches))
        if len(matches) < min_matches:
            return None
        return sorted(matches, key=lambda x: x.distance)[:max_matches]

    def _matches2pts(self, matches, keys):
        p1 = np.array([self.keys[m.queryIdx].pt for m in matches], dtype=np.float32)
        p2 = np.array([keys[m.trainIdx].pt for m in matches], dtype=np.float32)
        return p1, p2

    def _align(self, img):
        pts, descs = self._apply_sift(img)
        matches = self._match_descs(descs)
        if matches is None:
            return None
        ref_pts, img_pts = self._matches2pts(matches, pts)
        H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)
        h, w, c = self.img.shape
        return cv2.warpPerspective(img, H, (w, h))

    def _is_visible(self, img, threshold=6):
        # horizontal mean
        mu = img.mean(axis=1)
        # standard deviation over rows
        diff = mu.std(axis=0)
        # green color channel
        diff = diff[1]
        print("Vertical SD:", round(diff))
        return diff > threshold

    def analyze(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        aligned_img = self._align(img)
        if aligned_img is None:
            return "no-test"
        t = aligned_img[self.t_crop]
        c = aligned_img[self.c_crop]
        if not self._is_visible(t):
            return "invalid"
        elif self._is_visible(c):
            return "positive"
        else:
            return "negative"