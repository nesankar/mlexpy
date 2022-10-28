import pytest


@pytest.fixture()
def tester():
    return 1


def test_hello(tester):
    """Simple test to test that everything is working here in setup."""

    assert tester == 1
