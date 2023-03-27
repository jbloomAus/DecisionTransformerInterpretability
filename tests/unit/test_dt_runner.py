import pytest

from src.decision_transformer.utils import get_max_len_from_model_type


def test_get_max_len_from_model_type_dt():
    assert get_max_len_from_model_type("decision_transformer", 2) == 1
    assert get_max_len_from_model_type("decision_transformer", 3) == 2
    assert get_max_len_from_model_type("decision_transformer", 4) == 2
    assert get_max_len_from_model_type("decision_transformer", 5) == 2
    assert get_max_len_from_model_type("decision_transformer", 6) == 3
    assert get_max_len_from_model_type("decision_transformer", 7) == 3
    assert get_max_len_from_model_type("decision_transformer", 8) == 3
    assert get_max_len_from_model_type("decision_transformer", 9) == 4
    assert get_max_len_from_model_type("decision_transformer", 10) == 4
    assert get_max_len_from_model_type("decision_transformer", 11) == 4
    assert get_max_len_from_model_type("decision_transformer", 12) == 5
    assert get_max_len_from_model_type("decision_transformer", 13) == 5


def test_get_max_len_from_model_type_bc():
    assert get_max_len_from_model_type("clone_transformer", 1) == 1
    assert get_max_len_from_model_type("clone_transformer", 2) == 2
    assert get_max_len_from_model_type("clone_transformer", 3) == 2
    assert get_max_len_from_model_type("clone_transformer", 4) == 3
    assert get_max_len_from_model_type("clone_transformer", 5) == 3
    assert get_max_len_from_model_type("clone_transformer", 6) == 4
    assert get_max_len_from_model_type("clone_transformer", 7) == 4
    assert get_max_len_from_model_type("clone_transformer", 8) == 5
    assert get_max_len_from_model_type("clone_transformer", 9) == 5
    assert get_max_len_from_model_type("clone_transformer", 10) == 6
    assert get_max_len_from_model_type("clone_transformer", 11) == 6
    assert get_max_len_from_model_type("clone_transformer", 12) == 7
    assert get_max_len_from_model_type("clone_transformer", 13) == 7
