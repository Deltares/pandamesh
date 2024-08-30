import re
import textwrap

import pytest

from pandamesh.enum_base import FlexibleEnum, _show_options


class Color(FlexibleEnum):
    RED = 1
    GREEN = 2
    BLUE = 3


def test_show_options():
    _show_options(Color) == textwrap.dedent(
        """
        * RED
        * GREEN
        * BLUE
    """
    )


def test_check_options():
    assert Color(1) == Color.RED
    assert Color(Color.RED) == Color.RED
    assert Color.from_value(Color.RED) == Color.RED
    assert Color.from_value("RED") == Color.RED
    expected = (
        "'YELLOW' is not a valid Color. Valid options are:\n * RED\n * GREEN\n * BLUE"
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        Color.from_value("YELLOW")

    expected = expected.replace("'YELLOW'", "0")
    with pytest.raises(ValueError, match=re.escape(expected)):
        Color.from_value(0)
