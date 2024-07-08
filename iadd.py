import copy
import operator
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import MutableMapping, MutableSet
from typing import Iterable, Optional, TypeVar, Union

try:
    from typing import Protocol, runtime_checkable  # type: ignore
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

import numpy

T = TypeVar("T")


@runtime_checkable
class Addable(Protocol):
    def __add__(self: T, other: T) -> T:
        ...

Accumulatable = Union[Addable, MutableSet, MutableMapping]

def iadd(a: Accumulatable, b: Accumulatable) -> Accumulatable:
    """Add two accumulatables together, assuming the first is mutable"""
    if isinstance(a, Addable) and isinstance(b, Addable):
        return operator.iadd(a, b)
    elif isinstance(a, MutableSet) and isinstance(b, MutableSet):
        return operator.ior(a, b)
    elif isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
        if not isinstance(b, type(a)):
            raise ValueError(
                f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
            )
        lhs, rhs = set(a), set(b)
        # Keep the order of elements as far as possible
        for key in a:
            if key in rhs:
                a[key] = iadd(a[key], b[key])
        for key in b:
            if key not in lhs:
                a[key] = copy.deepcopy(b[key])
        return a
    raise ValueError(
        f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})"
    )

