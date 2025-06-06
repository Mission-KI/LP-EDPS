from edps.filewriter import _setup_matplotlib_if_needed


def test_multi_call_setup_matplotlib():
    _setup_matplotlib_if_needed()
    _setup_matplotlib_if_needed()
