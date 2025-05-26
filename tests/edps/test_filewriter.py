from edps.filewriter import setup_matplotlib


def test_multi_call_setup_matplotlib():
    setup_matplotlib()
    setup_matplotlib()
