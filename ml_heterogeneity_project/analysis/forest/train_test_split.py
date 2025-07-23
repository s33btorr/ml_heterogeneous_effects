from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np

# esta funcion usa train_test_split para dividir la data y ya transformarla a numpy (quiza deberia ser otra
#funcion, no se), y tambien hace el round pq con la data normalizada, no funciona sin eso

def split_data(treatment, outcome, covariates, percentage_test): # quite question pq la impresion ya no esta, pero quiza viene bien para evaluar si anda todo bien
    """ Splits Data into Test and Train.

    Args:
        treatment (column): Column that represents the treatment.

        outcome (column): Column that represents the outcome.

        x_cov (DataFrame): DataFrame of all columns representing the covariates.

        percentage_test (float): Percentage of the test sample.

    Returns:
        x_train, x_test, d_train, d_test, y_train, y_test (DataFrames):
        It returns all the sample already splitted.
    """

    d = treatment.astype(int)
    y = outcome
    x_cov = covariates # dataframe pq son muchas
    
    x_train, x_test, d_train, d_test, y_train, y_test = train_test_split(
        x_cov,
        d, 
        y, 
        test_size=percentage_test, 
        stratify=y, 
        random_state= 2000 # el random state tambien puede ser siempre el mismo (creo), por lo que podriamos ponerlo con SCR y simlplemente llamarlo. habria que agregarlo a la funcion claro.
        )

    return x_train, x_test, d_train, d_test, y_train, y_test

    """
    # en este caso, creo que seria mas que un print, un test que compruebe que las proporciones no estan tan alejadas unas de otras.
    print(f'Question {question}')
    original_proportion = y.value_counts(normalize=True)
    original_treatment_proportion = d.value_counts(normalize=True)
    print("Original proportions:")
    print(original_proportion, original_treatment_proportion)

    train_proportion = y_train.value_counts(normalize=True)
    train_treatment_proportion = d_train.value_counts(normalize=True)
    print("\nTrain proportions:")
    print(train_proportion, train_treatment_proportion)

    test_proportion = y_test.value_counts(normalize=True)
    test_treatment_proportion = d_test.value_counts(normalize=True)
    print("\nTest proportions:")
    print(test_proportion, test_treatment_proportion)
    """





if __name__ == "__main__":

    from normalizing_df import normalize_data

    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_cleaned.csv") 

    data_done = data_ready.copy()
    data_done = normalize_data(data_done)

    y_1 = data_1["Q1_1"]
    z_1 = data_1["Q1_1_treat"].astype(int)

    y_2 = data_1["Q1_2"]
    z_2 = data_1["Q1_2_treat"].astype(int)

    y_3 = data_1["Q2_1"]
    z_3 = data_1["Q2_1_treat"].astype(int)

    y_4 = data_1["Q2_2"]
    z_4 = data_1["Q2_2_treat"].astype(int)

    talk_list_1 = ["open_to_experience", "PC1", "empathic_concern_score", "Altruism", 
                "Positive_Reciprocity", "Negative_Reciprocity", "Trust", "Risk_Preferences",
                "rationality_score", "optimism_bias", "three_tax", "three_ban",
                "education", "age", "financialwellbeing", "born_in_lux"]

    x_cov = data_1[talk_list_1]

    outcome_list = [y_1, y_2, y_3, y_4]
    d_var_list = [z_1, z_2, z_3, z_4]

    for i,j in zip(d_var_list, outcome_list):
        split_data(i, j, x_cov, 0.4)

