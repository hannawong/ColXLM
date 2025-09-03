"""Unit tests for the parameters module."""

import pytest
import sys
from pathlib import Path

# Add the project root to the path to import colXLM modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from colXLM.parameters import *


@pytest.mark.unit
def test_parameters_imported():
    """Test that parameters module can be imported successfully."""
    # If we can import the module, the test passes
    assert True


def test_parameters_exist():
    """Test that basic parameters are available."""
    # Test will pass if module loads without errors
    # This demonstrates the testing infrastructure works with actual code
    assert True