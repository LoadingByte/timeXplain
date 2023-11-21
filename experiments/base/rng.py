import numpy as np


def reproducible_rng():
    rng = np.random.RandomState(1337)

    samples = rng.randint(0, 100_000, 99)
    expected = [3223, 57533, 99164, 9448, 85159, 87513, 22874, 71538, 28754,
                91220, 47176, 45321, 20870, 47542, 19802, 80536, 40065, 4891,
                60509, 95254, 49522, 24169, 25803, 77633, 30780, 32471, 37523,
                88814, 8529, 55899, 98708, 40456, 77384, 13885, 85575, 86650,
                67419, 95881, 94916, 62175, 47073, 44364, 66438, 90100, 30066,
                42485, 95150, 5879, 16148, 42514, 39706, 97897, 81570, 41668,
                39675, 14670, 59523, 57176, 2653, 11912, 81199, 22404, 42723,
                74514, 93373, 74249, 54364, 31053, 7986, 13388, 69933, 72120,
                31149, 98048, 38961, 77908, 4607, 10541, 16098, 20464, 33691,
                99836, 20679, 58644, 82550, 17516, 23219, 76276, 64722, 41605,
                62185, 17328, 55389, 78450, 8531, 47750, 19082, 98541, 69002]
    assert np.array_equal(samples, expected), "Random number generator behaves unexpectedly"

    return rng
