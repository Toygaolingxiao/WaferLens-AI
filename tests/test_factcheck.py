"""The fact-check guard: matching numbers pass, invented numbers fail."""

from waferlens.agent.factcheck import check_numbers


def test_matching_numbers_pass():
    result = check_numbers({"macro_f1": 0.81}, {"macro_f1": 0.8105})
    assert result.ok


def test_wrong_number_fails():
    result = check_numbers({"macro_f1": 0.95}, {"macro_f1": 0.81})
    assert not result.ok
    assert result.mismatches


def test_uncomputed_number_fails():
    result = check_numbers({"accuracy": 0.99}, {"macro_f1": 0.81})
    assert not result.ok
