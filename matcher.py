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
        sift = cv2.SIFT_create()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts, descs = sift.detectAndCompute(img, None)
        return pts, descs

    def _match_descs(self, descs, min_matches=50, max_matches=150, ratio=0.75):
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(self.descs, descs, k=2)
        matches = [m for m, n in matches if m.distance < ratio * n.distance]
        if len(matches) < min_matches:
            return None
        return sorted(matches, key=lambda x: x.distance)[:max_matches]

    def _align(self, img):
        pts, descs = self._apply_sift(img)
        matches = self._match_descs(descs)
        if matches is None:
            return None, None
        ref_pts = [self.keys[m.queryIdx].pt for m in matches]
        ref_pts = np.array(ref_pts, dtype=np.float32)
        img_pts = [pts[m.trainIdx].pt for m in matches]
        img_pts = np.array(img_pts, dtype=np.float32)
        H, _ = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)
        h, w = self.img.shape[0], self.img.shape[1]
        return cv2.warpPerspective(img, H, (w, h)), H

    def _is_visible(self, img, threshold=6):
        diff = img.mean(axis=1).std(axis=0)[1]
        return diff > threshold

    def _draw_detection(self, img, H):
        H_inv = np.linalg.inv(H)
        h, w = self.img.shape[1], self.img.shape[0]
        ref_pts = np.float32([[[0, 0], [0, w - 1], [h - 1, w - 1], [h - 1, 0]]])
        tra_pts = cv2.perspectiveTransform(ref_pts, H_inv)[0]
        for i in range(-1, 3):
            pt1 = (tra_pts[i][0], tra_pts[i][1])
            pt2 = (tra_pts[i + 1][0], tra_pts[i + 1][1])
            img = cv2.line(img, pt1, pt2, (0, 255, 0), 5)
        return img

    def analyze(self, img):
        aligned_img, H = self._align(img)
        if aligned_img is None:
            return "no-test", img
        t = aligned_img[self.t_crop]
        c = aligned_img[self.c_crop]
        img = self._draw_detection(img, H)
        if not self._is_visible(t):
            return "invalid", img
        elif self._is_visible(c):
            return "positive", img
        else:
            return "negative", img