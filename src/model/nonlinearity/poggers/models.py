from typing import TypeVar

import numpy as np


T = TypeVar("T")
def sub_nl(mu: T, alpha, nbx) -> T:
    return nbx * (np.sqrt(4 * (mu / nbx) * alpha + 1) - 1) / (2 * alpha)