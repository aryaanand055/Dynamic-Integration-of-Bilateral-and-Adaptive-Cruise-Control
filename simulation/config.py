import os
from pathlib import Path

# Project root directory discovery (portable across OS and invocation directories)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Standard Directory Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = PROJECT_ROOT / "plots"
DOCS_DIR = PROJECT_ROOT / "docs"
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard_logs"

# Ensure runtime output directories exist
for directory in (LOGS_DIR, PLOTS_DIR, MODELS_DIR, TENSORBOARD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Default Simulation Physics & Kinematic Parameters
DEFAULT_SIM_PARAMS = {
    'car_number': 15,
    'kd': 0.9,
    'kv': 0.5,
    'kc': 0.4,
    'v_des': 15.0,
    'max_v': 30.0,
    'min_v': 0.0,
    'min_dis': 6.0,
    'reaction_time': 0.8,
    'headway_time': 1.0,
    'max_a': 3.0,
    'min_a': -5.0,
    'min_gap': 2.0,
    'dt': 0.1,
    'sensor_delay': 0.0,
    'sensor_noise': 0.0,
}

def resolve_model_path(model_identifier: str = "td3_accel_agent.zip") -> str:
    """
    Resolves a model filename or relative path to a valid absolute path.
    Checks PROJECT_ROOT, MODELS_DIR, and current working directory.
    """
    identifier_path = Path(model_identifier)
    if identifier_path.is_absolute() and identifier_path.exists():
        return str(identifier_path)
    
    # Check within models directory first
    candidate = MODELS_DIR / model_identifier
    if candidate.exists():
        return str(candidate)
        
    # Check within project root
    candidate = PROJECT_ROOT / model_identifier
    if candidate.exists():
        return str(candidate)
        
    # If .zip was omitted
    if not model_identifier.endswith(".zip"):
        candidate_zip = MODELS_DIR / f"{model_identifier}.zip"
        if candidate_zip.exists():
            return str(candidate_zip)
            
    # Default fallback to models/
    return str(MODELS_DIR / model_identifier)
