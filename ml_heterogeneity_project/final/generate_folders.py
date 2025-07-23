from pathlib import Path

SRC = Path(__file__).parent.parent.resolve()
data_path = SRC / "data"
BLD = SRC / "bld" 

data_path.mkdir(parents=True, exist_ok=True)
BLD.mkdir(parents=True, exist_ok=True)


# After running this file, a folder call "bld" and a folder call "data" should be created.
# You should upload the .dta in the data folder to continue running the project.