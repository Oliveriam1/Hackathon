import math
import unittest

import numpy as np

from src.tracker import CONFIRM_HITS, MAX_CIRCLE_ONLY, MAX_MISSED
from src.vision import Vision
from tests.synthetic_scene import build_world, camera_homography, flight, project, render
from tests.test_rectangle_target import scene

WORLDS = {}


def world(style):
    if style not in WORLDS:
        WORLDS[style] = build_world(style)
    return WORLDS[style]


def random_view(tilt, seed, look_at=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    homography = camera_homography(rng.uniform(4, 7), tilt, rng.uniform(0, 360), rng.uniform(-20, 20),
                                   (look_at[0] + rng.uniform(-.3, .3), look_at[1] + rng.uniform(-.3, .3)))
    degrade = dict(blur=rng.uniform(0.5, 1.2), noise=rng.uniform(3, 7), gain=rng.uniform(0.8, 1.1), seed=seed)
    return homography, degrade


class PerspectiveTests(unittest.TestCase):
    def test_oblique_views_of_all_target_styles(self):
        for style in ('sheet', 'outline', 'color'):
            for tilt in (0, 30, 45, 55):
                hits = 0
                for seed in range(3):
                    homography, degrade = random_view(tilt, seed)
                    observation = Vision().observe(render(world(style), homography, **degrade))
                    truth = project(homography, (0, 0))
                    distances = [math.dist((c.x, c.y), truth) for c in observation.circles]
                    with self.subTest(style=style, tilt=tilt, seed=seed):
                        self.assertTrue(all(d < 6 for d in distances), 'falešný kandidát')
                    hits += any(d < 6 for d in distances)
                with self.subTest(style=style, tilt=tilt):
                    self.assertGreaterEqual(hits, 3 if tilt <= 45 else 2)

    def test_decoys_are_never_candidates(self):
        # Samotné kolečko, prázdný list, čtverec v listu, samotný prstenec.
        for decoy in ((-2.0, -1.4), (2.0, -1.4), (-2.0, 1.2), (2.0, 1.1)):
            for tilt in (0, 30, 50):
                homography, degrade = random_view(tilt, 7, decoy)
                observation = Vision().observe(render(world('sheet'), homography, **degrade))
                truth = project(homography, (0, 0))
                with self.subTest(decoy=decoy, tilt=tilt):
                    self.assertTrue(all(math.dist((c.x, c.y), truth) < 6 for c in observation.circles))

    def test_flight_keeps_lock_without_interruption(self):
        # Začátek průletu je šikmo z výšky: čtverec v listu návnady je tam
        # stejně velký jako kolečko. Každý 12. snímek je výpadek kamery (černý).
        vision = Vision()
        first = None
        for index, (homography, degrade) in enumerate(flight(48, seed=1)):
            dropout = first is not None and index % 12 == 11
            frame = render(world('sheet'), homography, **degrade)
            observation = vision.observe(np.zeros_like(frame) if dropout else frame)
            truth = project(homography, (0, 0))
            with self.subTest(frame=index):
                self.assertTrue(all(math.dist((c.x, c.y), truth) < 6 for c in observation.circles), 'falešný kandidát')
            if first is None and observation.confirmed:
                first = index
            if first is not None:
                with self.subTest(frame=index):
                    self.assertTrue(observation.confirmed)
                    self.assertEqual(observation.measured, not dropout)
                    # Predikce nezná náhodný roztřes pohledu během výpadku.
                    self.assertLess(math.dist((observation.target.x, observation.target.y), truth),
                                    25 if dropout else 10)
        self.assertIsNotNone(first)
        self.assertLess(first, 15)


class TrackingTests(unittest.TestCase):
    def confirmed_vision(self):
        vision = Vision()
        for _ in range(CONFIRM_HITS):
            observation = vision.observe(scene())
        self.assertTrue(observation.confirmed)
        return vision

    def test_short_dropout_is_bridged_by_prediction(self):
        vision = self.confirmed_vision()
        blank = np.zeros_like(scene())
        for index in range(MAX_MISSED):
            observation = vision.observe(blank)
            self.assertTrue(observation.confirmed)
            self.assertFalse(observation.measured)
            self.assertAlmostEqual(observation.target.x, 170, delta=3)
        self.assertFalse(vision.observe(blank).confirmed)
        self.assertFalse(vision.observe(scene()).confirmed)

    def test_circle_without_rectangle_only_holds_confirmed_target_briefly(self):
        vision = self.confirmed_vision()
        for _ in range(MAX_CIRCLE_ONLY):
            observation = vision.observe(scene(rectangle=False))
            self.assertTrue(observation.confirmed)
            self.assertTrue(observation.measured)
            self.assertIn('BEZ OBDELNIKU', observation.status)
        for _ in range(MAX_MISSED + 1):
            observation = vision.observe(scene(rectangle=False))
        self.assertFalse(observation.confirmed)
        # Samotné kolečko nikdy nezačne nový cíl.
        for _ in range(2 * CONFIRM_HITS):
            self.assertFalse(vision.observe(scene(rectangle=False)).confirmed)

    def test_offset_from_image_center(self):
        observation = self.confirmed_vision().observe(scene())
        self.assertAlmostEqual(observation.offset[0], (170-250)/250, delta=0.02)
        self.assertAlmostEqual(observation.offset[1], (145-150)/150, delta=0.02)


if __name__ == '__main__':
    unittest.main()
