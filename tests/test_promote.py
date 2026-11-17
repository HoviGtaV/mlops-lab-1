import math

from src.food11.promote import decide, standard_error


def test_standard_error():
    assert math.isclose(standard_error(0.75, 1000), math.sqrt(0.75 * 0.25 / 1000))


def test_small_gain_is_not_enough():
    # se ~ 0.013 at p=0.75, n=1096, so +1 point is within noise
    promote, threshold = decide(0.76, 0.75, 1096)
    assert not promote
    assert 0.775 < threshold < 0.78


def test_clear_gain_is_promoted():
    promote, _ = decide(0.80, 0.75, 1096)
    assert promote
