import pytest
from nanochat.gpt import GPTConfig


@pytest.fixture
def gpt_config():
    """Fixture providing a GPTConfig instance with default values."""
    return GPTConfig
