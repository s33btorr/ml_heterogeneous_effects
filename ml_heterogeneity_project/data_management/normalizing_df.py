import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path



def normalize_data(df):
    """ Normalizes the data frame, converting the values bewteen 0 and 1. """

    data_normalizada = df.copy()
    num_cols = data_normalizada.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    data_normalizada[num_cols] = scaler.fit_transform(data_normalizada[num_cols])
    return data_normalizada


if __name__ == "__main__":

    from pathlib import Path
    from clean_data import clean_dataset

    SRC = Path(__file__).parent.parent.resolve()
    data_path = SRC / "data" / "waves123_augmented_consent.dta"
    BLD = SRC / "bld" 
    waves_data = pd.read_stata(data_path, convert_categoricals=False)

    data_after_cleaning = clean_dataset(waves_data)
    normalize_data(data_after_cleaning).to_csv(BLD / "data_normalized.csv")
    

# Cosas que pueden fallar en esta funcion:
    # que se cambien columnas sin nan
    #que se cambien columnas con letras
    # columnas numericas no se estan normalizando