import pandas as pd
import numpy as np


def clean_dataset(df):
    """ Cleans the whole dataset using the helper functions below. """

    df_copy = df.copy()
    df_copy = eliminate_suspicious(df_copy)
    df_copy = generating_optimism_bias(df_copy)
    df_copy = arranging_other_variables(df_copy)

    dict_of_binary_bigger_equal = {
        'meat_w1': 4.475,
        'housing_w1': 20.09,
        'mobility_hyp_w1': 26.52
    }

    dict_of_binary_bigger = {
        'bonusdonation_w1': 0.297,
        'donate_lottery_w1': 0.198,
        'scale_sufficiency': 55.08,
        'EPC_ABC': 0.2259
    }

    df_copy = creating_binary_variables(df_copy, dict_of_binary_bigger_equal, dict_of_binary_bigger)

    df_copy = creating_big_five(df_copy)
    df_copy = recode_empathy(df_copy)

    df_copy = df_copy.rename(columns={
        'procrastinate': 'ProcrastinationExAnte', #Procrastination
        'timepreferences': 'ProcrastinationExPost', #Procrastination
        'selfcons_1': 'Social_Anxiety', #Self-consciousness
        'selfcons_2': 'Public_SelfConsciousness', #Self-consciousness
        'selfcons_3': 'Private_SelfConsciousness' #Self-consciousness
        })    # quite: , inplace=True    SI FALLA, agregar again

    df_copy = arrange_mariel_boatlift_treatment(df_copy)
    df_copy = generating_indexes(df_copy)

    return df_copy




def eliminate_suspicious(df):
    """ Eliminating people that did not finished and with suspicious answers. """

    df = df[
    (df['Finished_w3'] == 1) &
    (df['anysuspicious'] == 0)
    ]
    return df


def generating_optimism_bias(df):
    """ Generates variable of optimism bias. """

    df['conf_new'] = 0
    df.loc[df['confidence'] == 4, 'conf_new'] = 1
    df.loc[df['confidence'] == 3, 'conf_new'] = 2
    df.loc[df['confidence'] == 2, 'conf_new'] = 3
    df.loc[df['confidence'] == 1, 'conf_new'] = 4
    df.loc[df['confidence'] == 99, 'conf_new'] = 0
    df['conf_new'] = pd.to_numeric(df['conf_new'], errors='coerce') # revisar este codigo e intentar hacerlo de nuevo pq no creo que esta linea sea necesaria

    ranking_map = {
        "Jan-50": 1,
        "51-150": 2,
        "151-350": 3,
        "351-1026": 4
        }

    df['ranking_num'] = df['ranking_w2'].map(ranking_map).astype(int)


    df['optimism_bias'] = df['ranking_num'] - df['conf_new']

    return df


def arranging_other_variables(df):
    """ Arranging variables by multipliying and dividing by 100. """

    df['scale_sufficiency'] = df['scale_sufficiency']*100
    df['donate_lottery_w1'] = df['donate_lottery_w1']/100

    return df

def creating_binary_variables(df, dict_bigger_equal, dict_bigger):
    """ Generates binary variables, given a dictionary with name of variable
    and number for cut off. """

    for i, j in dict_bigger_equal.items():
        df[i] = np.where(df[i] >= j, 0, 1)

    for i, j in dict_bigger.items():
        df[i] = np.where(df[i] > j, 0, 1)

    return df    

def creating_big_five(df):
    """ Generating Big Five variables. """
    
    big_five_inverted_list = ['bigfive_1', 'bigfive_3', 'bigfive_4', 'bigfive_5', 'bigfive_7']

    for i in big_five_inverted_list:
        df[i] = 6 - df[i]

    df['openness'] = df[['bigfive_5', 'bigfive_10']].mean(axis=1)
    df['conscientiousness'] = df[['bigfive_3', 'bigfive_8']].mean(axis=1)
    df['extraversion'] = df[['bigfive_1', 'bigfive_6']].mean(axis=1)
    df['agreeableness'] = df[['bigfive_2', 'bigfive_7']].mean(axis=1)
    df['neuroticism'] = df[['bigfive_4', 'bigfive_9']].mean(axis=1)

    return df

def creating_rat_score_and_growthmind(df):
    """ Generating more variables. """

    df['rationality_score'] = (
        (df['rationality1'] == 2).astype(int) +
        (df['rationality2'] == 8).astype(int) +
        (df['rationality3'] == 50).astype(int)
        )

    df['growthmind'] = df['growthmind'].replace({1: 4, 2: 3, 3: 2, 4: 1}).astype(float)

    return df

def recode_empathy(df):
    """ Recodes empathy variables. """

    diccionario = {23: 4, 4: 3, 3: 2, 2: 1, 1: 0}
    lista_empathy = ["empathy_1", "empathy_2", "empathy_3", "empathy_4"]

    for col in lista_empathy:
        df[col] = df[col].apply(lambda x: diccionario.get(x, x))

    df['empathic_concern_score'] = df['empathy_1'] + df['empathy_2']
    df['perspective_taking_score'] = df['empathy_3'] + df['empathy_4']

    return df

def arrange_mariel_boatlift_treatment(df):
    """ Arranging all variables related to treatment of marielitos. """

    df['knowledge_mariel'] = (df['marielboatlift'] == 1).astype(int) # if answer was right, dummy==1

    # This could be easily done with a loop, but I am tired...

    df['Q1_1'] = df['question1treat_1'].fillna(df['question1control_1'])
    df['Q1_1_treat'] = np.where(df['question1treat_1'].notna(), 1, 0)

    df['Q1_2'] = df['question1treat_2'].fillna(df['question1control_2'])
    df['Q1_2_treat'] = np.where(df['question1treat_2'].notna(), 1, 0)

    df['Q2_1'] = df['question2treat_1'].fillna(df['question2control_1'])
    df['Q2_1_treat'] = np.where(df['question2treat_1'].notna(), 1, 0)

    df['Q2_2'] = df['question2treat_2'].fillna(df['question2control_2'])
    df['Q2_2_treat'] = np.where(df['question2treat_2'].notna(), 1, 0)

    return df

def generating_indexes(df):
    """ Generates indexes with different variables. """

    # This could be up for discussion, we do average
    list_of_var = [
        'trustscientists_1', 'trustscientists_2', 'trustscientists_3',
        'trustinst_1', 'trustinst_2', 'trustinst_3', 'trustinst_4']
    for i in list_of_var:
        df[i] = df[i].replace(np.NaN, df[i].mean())

    df["trust_in_science"] = (0.33*df["trustscientists_1"]) + (0.33*df["trustscientists_2"]) + (0.33*df["trustscientists_3"])
    df["trust_in_institutions"] = (0.25*df["trustinst_1"]) + (0.25*df["trustinst_2"]) + (0.25*df["trustinst_3"]) + (0.25*df["trustinst_4"])

    df["resource_food_wi"] = (0.5*df["worldissues_1"]) + (0.5*df["worldissues_8"])
    df["pandemic_risk_wi"] = df["worldissues_7"]
    df["rest_wi"] = (0.25*df["worldissues_2"]) + (0.25*df["worldissues_3"]) + (0.25*df["worldissues_4"]) + (0.25*df["worldissues_5"])
    df["migration_wi"] = df["worldissues_6"]

    df["energy_saving_ll"] = (0.2*df["longlistbeh_w1_1"]) + (0.2*df["longlistbeh_w1_2"])+ (0.2*df["longlistbeh_w1_3"])+ (0.2*df["longlistbeh_w1_4"])+ (0.2*df["longlistbeh_w1_5"])
    df["sustainable_food_ll"] = (0.25*df["longlistbeh_w1_6"]) + (0.25*df["longlistbeh_w1_7"]) + (0.25*df["longlistbeh_w1_8"]) + (0.25*df["longlistbeh_w1_10"])
    df["second_hand_ll"] = df["longlistbeh_w1_9"]

    df["three_tax"] = (0.33*df["carbontax1"]) + (0.33*df["meattax_w1"]) + (0.33*df["housingtax_w1"])

    df["PC1"] = 0.2*df["Public_SelfConsciousness"] + 0.2*df["Private_SelfConsciousness"]+ 0.2*df["neuroticism"]+ 0.2*df["agreeableness"]+ 0.2*df["Social_Anxiety"]
    df["open_to_experience"] = 0.33*df["openness"] + 0.33*df["extraversion"] + 0.33*df["growthmind"]
    df["three_ban"] = (0.33*df["carban_w1"]) + (0.33*df["meatban_w1"]) + (0.33*df["housingban_w1"])

    return df






if __name__ == "__main__":

    from pathlib import Path

    SRC = Path(__file__).parent.parent.resolve()
    data_path = SRC / "data" / "waves123_augmented_consent.dta"
    waves_data = pd.read_stata(data_path, convert_categoricals=False)

    data_after_cleaning = clean_dataset(waves_data)

    N = len(data_after_cleaning)
    print("Number of participants (rows):", N)

