"""The suite-wide fixture restores the recursion limit after every test."""

import sys

BEFORE = sys.getrecursionlimit()


def test_a_test_may_raise_the_recursion_limit():
    sys.setrecursionlimit(2**14)
    assert sys.getrecursionlimit() == 2**14


def test_the_next_test_sees_the_original_limit():
    assert sys.getrecursionlimit() == BEFORE
