import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from detector_neumonia import preprocess


def test_preprocess_shape():
    fake_img = np.random.randint(0, 255, (600, 600), dtype=np.uint8)
    result = preprocess(fake_img)
    assert result.shape == (1, 512, 512, 1)


def test_preprocess_range():
    fake_img = np.random.randint(0, 255, (600, 600), dtype=np.uint8)
    result = preprocess(fake_img)
    assert result.min() >= 0.0 and result.max() <= 1.0