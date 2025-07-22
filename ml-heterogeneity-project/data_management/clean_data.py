import pandas as pd
import numpy as np
import pandas as pd 
import numpy as np


def 

dataset = dataset[
    (dataset['Finished_w3'] == 1) & #re
    (dataset['anysuspicious'] == 0) #& #re
    #(dataset['WTA'] <= 250) & #re
    #(dataset['WTP'] <= 250) & #re
    #(dataset['gender'].notna()) #re
    ]

N = len(dataset)
print("Number of participants (rows):", N)


if __name__ == "__main__":

    from config import data_path

    waves_data = pd.read_stata(data_path, convert_categoricals=False)
