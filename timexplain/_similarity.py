from typing import Sequence

from scipy.spatial.distance import correlation as _correlation

from timexplain._utils import optional_njit


def correlation(impacts_1: Sequence[float], impacts_2: Sequence[float]) -> float:
    if _is_constant(impacts_1) or _is_constant(impacts_2):
        return 0
    else:
        return 1 - _correlation(impacts_1, impacts_2)


@optional_njit
def _is_constant(arr):
    first = arr[0]
    for v in arr[1:]:
        if v != first:
            return False
    return True
