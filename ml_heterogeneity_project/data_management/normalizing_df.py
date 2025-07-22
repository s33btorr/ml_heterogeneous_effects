import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path



def normalize_data(df):
    data_normalizada = df.copy()
    num_cols = data_normalizada.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    data_normalizada[num_cols] = scaler.fit_transform(data_normalizada[num_cols])
    return data_normalizada


if __name__ == "__main__":

    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_cleaned.csv") 

    data_done = data_ready.copy()
    #data_done = data_done[data_done['gender'].notna()]

    normalize_data(data_done).to_csv("data_normalized.csv")

# Cosas que pueden fallar en esta funcion:
    # que se cambien columnas sin nan
    #que se cambien columnas con letras
    # columnas numericas no se estan normalizando