import pytest
from mlexpy.src.defaultordereddict import DefaultOrderedDict


def test_initial_value():
    dod = DefaultOrderedDict(lambda: [])

    # Test that if there is no key provided, the default value is correct.
    assert dod[0] == []

    dod = DefaultOrderedDict(lambda: 0)
    # Test that if there is no key provided, the default value is correct.
    assert dod[0] == 0


def test_correct_ordering():
    dod = DefaultOrderedDict(lambda: 0)
    keys = ["6", "c", "a", "d", "b"]

    for item in keys:
        dod[item] += 1

    dod_keys = list(dod.keys())

    # Test that the ordering in the dict is the same as when input into the dict.
    assert all([kv == dod_keys[i] for i, kv in enumerate(keys)])

    # Test that the values are correct as we retrieve them

    assert dod["6"] == 1
    assert dod["z"] == 0
