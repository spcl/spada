from collections import defaultdict
from typing import Generic, TypeVar


T = TypeVar('T')


class Versioning(Generic[T]):
    # Mapping from variable names to the number of fields allocated for that variable
    # Used to generate unique names for variables
    _var_counter: dict[str, int]

    def __init__(self, cls: T):
        self._var_counter = defaultdict(int)
        self.cls = cls

    def next_version(self, name: str) -> T:
        """
        Gets the next version of a variable name.
        """
        version = self._var_counter[name]
        self._var_counter[name] += 1
        return self.cls(name, version)
