import unittest
import cv2
import numpy as np
from src.circle_detector import CircleDetector
from src.vision import Vision


class CircleTests(unittest.TestCase):
    def test_disks_and_rings(self):
        for thickness in (-1, 4):
            for color in ((255, 255, 255), (0, 0, 255), (0, 255, 0)):
                with self.subTest(thickness=thickness, color=color):
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.circle(frame, (160, 120), 35, color, thickness)
                    circles = CircleDetector().detect(frame)
                    self.assertEqual(len(circles), 1)
                    self.assertAlmostEqual(circles[0].x, 160, delta=2)
                    self.assertAlmostEqual(circles[0].y, 120, delta=2)

    def test_reject_other_shapes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 30), (130, 130), (255, 255, 255), -1)
        cv2.ellipse(frame, (300, 100), (65, 30), 0, 0, 360, (255, 255, 255), -1)
        cv2.fillPoly(frame, [np.array([[40, 300], [140, 300], [90, 200]])], (255, 255, 255))
        cv2.line(frame, (220, 300), (500, 350), (255, 255, 255), 8)
        cv2.circle(frame, (635, 460), 35, (255, 255, 255), -1)
        self.assertEqual(CircleDetector().detect(frame), [])

    def test_multiple_circles_without_green_reference(self):
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        cv2.circle(frame, (80, 120), 25, (255, 255, 255), -1)
        cv2.circle(frame, (300, 120), 40, (255, 255, 255), -1)
        self.assertEqual(len(CircleDetector().detect(frame)), 2)
        self.assertEqual(Vision().observe(frame).circles, [])

    def test_blank(self):
        self.assertEqual(CircleDetector().detect(np.zeros((240, 320, 3), dtype=np.uint8)), [])

    def test_blurred_dim_circle(self):
        frame = np.full((240, 320, 3), 25, dtype=np.uint8)
        cv2.circle(frame, (160, 120), 40, (55, 55, 55), -1)
        frame = cv2.GaussianBlur(frame, (15, 15), 3)
        circles = CircleDetector().detect(frame)
        self.assertTrue(any(abs(c.x-160) < 5 and abs(c.y-120) < 5 for c in circles))

    def test_broken_ring(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.ellipse(frame, (160, 120), (40, 40), 0, 15, 345, (180, 180, 180), 3)
        circles = CircleDetector().detect(frame)
        self.assertTrue(any(abs(c.x-160) < 5 and abs(c.y-120) < 5 for c in circles))
