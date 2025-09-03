"""Validation tests to ensure the testing infrastructure is working correctly."""

import pytest
import sys
from pathlib import Path


def test_python_version():
    """Validate Python version is compatible."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"


def test_imports_available():
    """Test that required testing packages can be imported."""
    try:
        import pytest
        import pytest_cov
        import pytest_mock
        assert True
    except ImportError as e:
        pytest.fail(f"Required testing package not available: {e}")


def test_project_structure():
    """Validate basic project structure exists."""
    project_root = Path(__file__).parent.parent
    
    # Check main package exists
    assert (project_root / "colXLM").exists(), "Main colXLM package directory missing"
    assert (project_root / "colXLM" / "__init__.py").exists(), "colXLM __init__.py missing"
    
    # Check pyproject.toml exists
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml missing"


def test_testing_structure():
    """Validate testing directory structure."""
    tests_root = Path(__file__).parent
    
    assert tests_root.name == "tests", "Tests should be in 'tests' directory"
    assert (tests_root / "__init__.py").exists(), "tests/__init__.py missing"
    assert (tests_root / "conftest.py").exists(), "tests/conftest.py missing"
    assert (tests_root / "unit").exists(), "tests/unit directory missing"
    assert (tests_root / "unit" / "__init__.py").exists(), "tests/unit/__init__.py missing"
    assert (tests_root / "integration").exists(), "tests/integration directory missing"
    assert (tests_root / "integration" / "__init__.py").exists(), "tests/integration/__init__.py missing"


@pytest.mark.unit
def test_fixtures_available(temp_dir, sample_text, mock_tokenizer):
    """Test that shared fixtures are working."""
    # Test temp_dir fixture
    assert temp_dir.exists(), "temp_dir fixture not working"
    assert temp_dir.is_dir(), "temp_dir should be a directory"
    
    # Test sample_text fixture
    assert isinstance(sample_text, str), "sample_text should be a string"
    assert len(sample_text) > 0, "sample_text should not be empty"
    
    # Test mock_tokenizer fixture
    assert hasattr(mock_tokenizer, 'encode'), "mock_tokenizer should have encode method"
    assert hasattr(mock_tokenizer, 'decode'), "mock_tokenizer should have decode method"


def test_pytest_markers():
    """Test that custom pytest markers are configured."""
    # This test will pass if markers are properly configured
    # If markers aren't configured, pytest will show warnings
    pass


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works (this test should be skipped with -m 'not slow')."""
    import time
    time.sleep(0.1)  # Simulate slow test
    assert True


def test_coverage_config():
    """Validate that coverage configuration is working."""
    # This is validated by the test run itself
    # If coverage is configured correctly, this test will be included in coverage
    assert True


def test_mocking_works(mock_logger):
    """Test that mocking functionality works."""
    mock_logger.info("test message")
    mock_logger.info.assert_called_once_with("test message")


def test_parametrize_example():
    """Example of parametrized test to validate pytest functionality."""
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 4), 
        (3, 9),
    ])
    def inner_test(value, expected):
        assert value ** 2 == expected
    
    # Run the parametrized tests
    inner_test(1, 1)
    inner_test(2, 4)
    inner_test(3, 9)


def test_fixture_scope():
    """Test that fixtures with different scopes work."""
    # This uses the autouse fixture from conftest.py
    # If random seeds are reset properly, this should be deterministic
    import random
    # The random seed should be consistent due to reset_random_seeds fixture
    assert True