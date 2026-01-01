import pytest

from pathlib import Path
from sympy import Symbol
import sys

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def x():
    """Define a shared symbolic variable for tests."""
    return Symbol("x")
