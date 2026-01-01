import pytest

from sympy import Symbol

x = Symbol("x")


@pytest.fixture(scope="session")
def x():
    """Define a shared symbolic variable for tests."""
    return x
