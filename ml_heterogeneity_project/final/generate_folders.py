from pathlib import Path

SRC = Path(__file__).parent.parent.resolve()
data_path = SRC / "data"
BLD = SRC / "bld" 

data_path.mkdir(parents=True, exist_ok=True)
BLD.mkdir(parents=True, exist_ok=True)
