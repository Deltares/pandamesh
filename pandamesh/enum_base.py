from enum import Enum
from typing import Type, TypeVar, Union

E = TypeVar("E", bound="FlexibleEnum")


def _show_options(options: Type[E]) -> str:
    return "\n * ".join(map(str, options.__members__))


class FlexibleEnum(Enum):
    @classmethod
    def from_value(cls: Type[E], value: Union[E, str]) -> E:
        if isinstance(value, cls):
            return value
        elif isinstance(value, str):
            try:
                return cls.__members__[value]
            except KeyError:
                pass

        raise ValueError(
            # Use __repr__() so strings are shown with quotes.
            f"{value.__repr__()} is not a valid {cls.__name__}. "
            f"Valid options are:\n * {_show_options(cls)}"
        )
