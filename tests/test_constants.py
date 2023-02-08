import pytest

from src.streamlit_app.constants import twenty_idx_format_func

# def twenty_idx_format_func(idx):
#     if idx < 10:
#         return IDX_TO_OBJECT[idx]
#     elif idx < 16:
#         return IDX_TO_COLOR[idx-10]
#     elif idx < 19:
#         return IDX_TO_STATE[idx-16]
#     else:
#         return idx

def test_twenty_idx_format_func():

    assert twenty_idx_format_func(0) == "unseen"
    assert twenty_idx_format_func(9) == "lava"
    assert twenty_idx_format_func(10) == "agent"
    assert twenty_idx_format_func(11) == "red"
    assert twenty_idx_format_func(16) == "grey"
    assert twenty_idx_format_func(17) == "open"
    assert twenty_idx_format_func(18) == "closed"
    assert twenty_idx_format_func(19) == "locked"
