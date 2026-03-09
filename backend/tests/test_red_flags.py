from app.red_flags import detect_red_flags


def test_detects_chest_pain():
    hits = detect_red_flags("I have chest pain and shortness of breath")
    codes = {h.code for h in hits}
    assert "cardiac_chest_pain" in codes
    assert "resp_distress" in codes


def test_no_hit_for_minor_cold_text():
    hits = detect_red_flags("Mild sore throat for 1 day, no fever")
    assert hits == []
