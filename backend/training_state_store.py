from enum import Enum
from typing import Dict

class TrainingState(str, Enum):
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"

# athlete_id -> state
_state: Dict[int, TrainingState] = {}
_run_count: Dict[int, int] = {}

def set_training(athlete_id: int):
    _state[athlete_id] = TrainingState.TRAINING

def set_ready(athlete_id: int, run_count: int):
    _state[athlete_id] = TrainingState.READY
    _run_count[athlete_id] = run_count

def set_failed(athlete_id: int):
    _state[athlete_id] = TrainingState.FAILED

def get_state(athlete_id: int):
    return _state.get(athlete_id)

def get_run_count(athlete_id: int):
    return _run_count.get(athlete_id, 0)
