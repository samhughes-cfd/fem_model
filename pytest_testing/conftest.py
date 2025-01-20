# pytest_testing/conftest.py

import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def create_logs_directory():
    """Ensure that the logs directory exists before tests run."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)