import pytest

from eval.calibration import auroc, render, split_scores, threshold_rows


def test_auroc_perfect_none_and_ties():
    assert auroc([5, 6], [1, 2]) == 1.0
    assert auroc([1, 2], [5, 6]) == 0.0
    assert auroc([1, 3], [2]) == 0.5  # one win, one loss
    assert auroc([2], [2]) == 0.5  # a tie counts half
    with pytest.raises(ValueError):
        auroc([], [1])


def data():
    def q(answerable, score):
        return {"answerable": answerable, "runs": {"m": {"top1_score": score}}}

    return {"questions": [q(True, 4.0), q(True, 1.0), q(True, 3.0), q(False, 2.0), q(False, 0.5), q(True, None)]}


def test_split_scores_ignores_missing_scores():
    assert split_scores(data(), "m") == ([4.0, 1.0, 3.0], [2.0, 0.5])


def test_threshold_rows_count_both_kinds_of_error():
    a, u = split_scores(data(), "m")
    rows = threshold_rows(a, u, [0.0, 1.5, 2.5, 5.0], 1)
    assert rows[0] == ["0.0", "0/2 (0%)", "0/3 (0%)"]
    assert rows[1] == ["1.5", "1/2 (50%)", "1/3 (33%)"]  # 0.5 refused; 1.0 wrongly refused
    assert rows[2] == ["2.5", "2/2 (100%)", "1/3 (33%)"]
    assert rows[3] == ["5.0", "2/2 (100%)", "3/3 (100%)"]


def test_render_contains_summary_and_threshold_table():
    text = render(data(), ["m"])
    assert "AUROC" in text and "| 0.83 |" in text  # 5 of 6 pairs won
    assert "unanswerable refused (correct)" in text
