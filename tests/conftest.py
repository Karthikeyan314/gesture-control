from pathlib import Path

import pytest

from gesture_control.config import load_config
from gesture_control.fixtures import Recording, load_recording

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PATHS = sorted(FIXTURES.glob("*.json"))


@pytest.fixture(scope="session")
def config():
    return load_config()


def recording(name: str) -> Recording:
    return load_recording(FIXTURES / f"{name}.json")


def first_hand(name: str):
    """The first frame of a fixture that has a hand in it."""
    return next(h for _, h in recording(name).frames if h is not None)
