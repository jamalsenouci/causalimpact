import os
import pytest

here = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def FIXTURES_FOLDER():
    return os.path.join(here, "fixtures")
