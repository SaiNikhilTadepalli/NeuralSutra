import pytest

from pathlib import Path
from sympy import Symbol
import sys

from neuralsutra.compiler import Compiler

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def x():
    """Define a shared symbolic variable for tests."""
    return Symbol("x")


@pytest.fixture
def compiler():
    """Define a shared Compiler instance for tests."""
    return Compiler("models/router.pth", "models/vocab.json")
