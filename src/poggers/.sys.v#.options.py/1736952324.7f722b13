from pathlib import Path
from dataclasses import dataclass

class Singleton(type):
    """A metaclass for creating singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Create an instance if it doesn't exist yet
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

@dataclass
class PoggerOptions(metaclass=Singleton):
    vdm_path: Path = Path("/brildata/vdmdata23/")