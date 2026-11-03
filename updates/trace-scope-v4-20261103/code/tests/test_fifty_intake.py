import pytest

from tools.evolution_fifty.intake import FIELDS, parse_cards


def card_text():
    return "\n\n".join(
        f"实验 {i:02d}｜title{i}\n" + "\n".join(f"{k}：value" for k in FIELDS)
        for i in range(1, 51)) + "\n建议论证图表组合\nend"


def test_all_fifty_mandatory_not_completed():
    cards = parse_cards(card_text())
    assert len(cards) == 50
    assert [c["id"] for c in cards] == [f"E{i:02d}" for i in range(1, 51)]
    assert all(c["required"] and c["status"] == "not_started" and not c["evidence"] for c in cards)
    assert cards[30]["dependencies"] == ["E06", "E07"]


def test_missing_card_or_field_rejected():
    with pytest.raises(ValueError, match="Exactly"):
        parse_cards(card_text().replace("实验 17｜", "missing17｜"))
    with pytest.raises(ValueError, match="lacks"):
        parse_cards(card_text().replace("最小设计：", "missing：", 1))
