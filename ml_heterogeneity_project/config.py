from pathlib import Path


# from ml-heterogeneity-project.config import BLD, numeric_cols

SRC = Path(__file__).parent.resolve()

data_path = SRC / "data" / "waves123_augmented_consent.dta"

BLD = ROOT.joinpath("bld").resolve()
DATA = ROOT.joinpath("data").resolve()


# Parameters for setting up the predictor
number_of_classes = 2
threshold = 0.5
number_of_detections_per_image = 200

# Columns that contains numbers
numeric_cols = [
    "paid_up_capital",
    "surp_and_prof",
    "deposits",
    "loans_and_discounts_stocks_and_securities",
    "cash_and_exchanges",
]