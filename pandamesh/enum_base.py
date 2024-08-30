from enum import Enum, EnumMeta
from typing import Any, Type, TypeVar, Union, cast

E = TypeVar("E", bound="FlexibleEnum")


def _show_options(options: Type[E]) -> str:
    return "\n * ".join(map(str, options.__members__))


class AttributeErrorMeta(EnumMeta):
    def __getattr__(cls, name: str) -> Any:
        try:
            return cls.__members__[name]
        except KeyError:
            raise AttributeError(
                f"{name} is not a valid {cls.__name__}. "
                f"Valid options are:\n * {_show_options(cls)}"
            )


class FlexibleEnum(Enum, metaclass=AttributeErrorMeta):
    @classmethod
    def parse(cls: Type[E], value: str) -> E:
        try:
            return cls.__members__[value]
        except KeyError:
            raise ValueError(
                # Use __repr__() so strings are shown with quotes.
                f"{value.__repr__()} is not a valid {cls.__name__}. "
                f"Valid options are:\n * {_show_options(cls)}"
            )

    @classmethod
    def from_value(cls: Type[E], value: Union[E, str]) -> E:
        if isinstance(value, cls):
            return value
        else:
            value = cast(str, value)
            return cls.parse(value)
