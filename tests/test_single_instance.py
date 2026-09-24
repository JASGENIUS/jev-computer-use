"""One owner per hotkey."""
from __future__ import annotations

import pytest

from jevflow import single_instance as si


def test_the_lock_is_keyed_on_the_hotkeys_not_the_program():
    """JevFlow on Win+Alt and Computer Use on Ctrl+Win must both be able to
    run. They only conflict if they grab the SAME keys."""
    assert si._name_for(["windows+alt"]) != si._name_for(["ctrl+windows"])


def test_the_same_keys_produce_the_same_lock_whatever_the_order():
    assert si._name_for(["windows+alt", "ctrl+windows"]) == \
           si._name_for(["ctrl+windows", "windows+alt"])


def test_case_and_spacing_do_not_create_a_second_lock():
    assert si._name_for([" Windows+Alt "]) == si._name_for(["windows+alt"])


def test_none_is_not_a_hotkey():
    assert si._name_for(["none", "windows+alt"]) == si._name_for(["windows+alt"])


def test_a_second_acquire_of_the_same_keys_is_refused():
    keys = ["jevflow+test+unique+keys"]
    assert si.acquire(keys) is True
    # Same process, but the mutex already exists - which is exactly what the
    # second copy sees.
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW(None, ctypes.c_bool(True),
                     ctypes.c_wchar_p(si._name_for(keys)))
    assert k32.GetLastError() == si.ERROR_ALREADY_EXISTS


def test_acquiring_different_keys_still_succeeds():
    assert si.acquire(["jevflow+test+other+keys"]) is True
