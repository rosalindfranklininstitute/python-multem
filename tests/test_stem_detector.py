import multem


def test_stem_detector():
    stem_detector = multem.STEMDetector()
    stem_detector.type = "Test"
    stem_detector.cir = [(0, 1), (2, 3)]
    stem_detector.radial = [(0, [1, 2, 3, 4]), (2, [5, 6, 8, 9])]
    stem_detector.matrix = [(3, [1, 2, 3, 4]), (4, [5, 6, 7, 8])]

    assert stem_detector.type == "Test"
    assert stem_detector.cir == [(0, 1), (2, 3)]
    assert stem_detector.radial == [(0, [1, 2, 3, 4]), (2, [5, 6, 8, 9])]
    assert stem_detector.matrix == [(3, [1, 2, 3, 4]), (4, [5, 6, 7, 8])]
