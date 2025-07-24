from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
import pandas as pd 

from analysis.forest.causal_forest import generating_causal_forest
from analysis.forest.train_test_split import split_data

def tuning_causal_forest(n_estimators_grid, min_samples_leaf_grid, subsample_fr_grid,
                        treatment, outcome, covariates, percentage_test,
                        model_outcome, model_treatment, random_seed):

    """ Tunnes the Causal Forest. Given the list of parameters it gives the "Best" 
    Results when comparing the y_pred (the outcome predicted using training sample)
    with the y_test.
    """
    results = []

    for n_est in n_estimators_grid:
        for min_leaf in min_samples_leaf_grid:
            for subsample in subsample_fr_grid:

                x_train, x_test, d_train, d_test, y_train, y_test = split_data(treatment, outcome, covariates, percentage_test)
                model = generating_causal_forest(model_outcome, model_treatment, n_est, min_leaf, subsample, random_seed, y_train, d_train, x_train)

                te_pred = model.effect(x_test)

                # Como no tenemos el efecto verdadero, usamos proxy: error entre outcome real y predicho
                # Esto no es perfecto, pero sirve como criterio relativo
                y_pred = model.const_marginal_effect(x_test) #* z_test_1  # Aprox. Chat me dice que deberia multiplicarlo por el z_test, pero realmente no entiendo pq...
                mse = mean_squared_error(y_test, y_pred)

                results.append({
                    'n_estimators': n_est,
                    'min_samples_leaf': min_leaf,
                    'max_samples': subsample,
                    'MSE_proxy': mse
                })

    # Mostrar resultados ordenados por menor error
    results_df = pd.DataFrame(results)
    best_result = results_df.sort_values(by='MSE_proxy').iloc[0]

    print("Lowest MSE:")
    print(best_result)