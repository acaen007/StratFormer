from stratformer.novelty.novelty_detector import NoveltyDetector


def test_novelty_triggers_when_sustained_low():
    det = NoveltyDetector(window_size=3, threshold=0.5)
    assert det.update(0.6) is False
    assert det.update(0.4) is False
    assert det.update(0.49) is True
    # Any high value resets window behavior by overwriting older values
    assert det.update(0.9) is False


