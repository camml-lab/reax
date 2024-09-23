from jax import config, random
import pytest

import reax.loggers.pandas

# Some numerical test rely on checking against numpy implementations and np will default to 64-bit,
# so do that too
config.update("jax_enable_x64", True)


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


@pytest.fixture
def default_logger(tmp_path) -> reax.loggers.pandas.PandasLogger:
    """A default logger that can be used for testing"""
    return reax.loggers.pandas.PandasLogger(tmp_path)
