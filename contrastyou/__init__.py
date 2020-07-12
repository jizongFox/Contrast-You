from pathlib import Path

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))
