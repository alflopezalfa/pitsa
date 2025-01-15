import numpy as np
import pandas as pd
import pitsa

def run_sbapc(df, 
              base_model, 
              correction_model, 
              n_train, 
              time_starts=None, 
              correction_windows=None, 
              forecast_lenght=0):
    """
    Run the Sequential BAPC.

    Parameters:
    df (DataFrame): Input data.
    base_model (object): The base model to be used.
    correction_model (object): The correction model to be used.
    n_train (int): Number of training samples.
    time_starts (list, optional): List of starting time steps. Defaults to None.
    correction_windows (list, optional): List of correction window periods. Defaults to None.
    forecast_lenght (int, optional): Number of future time steps to forecast. Defaults to 0.

    Returns:
    tuple: Parameters DataFrame, explainers dictionary, and LIME coefficients DataFrame.
    """
    
    # Determine the time steps to start the training
    if time_starts is None:
        n_samples = df.shape[0]
        first_time_start = 0
        last_time_start = n_samples - n_train
        time_starts = range(first_time_start, last_time_start+1)
    else:
        time_starts = np.atleast_1d(time_starts)

    # Determine the correction window periods
    if correction_windows is None:
        n_samples = df.shape[0]
        correction_window_start = 0
        correction_window_end = n_train #+ 1
        correction_windows = range(correction_window_start, correction_window_end+1)

    # Initialize the explainer
    explainer = pitsa.xai.bapc.Explainer(
        base_model=base_model, 
        correction_model=correction_model
    )

    explainers = {}
    data = []

    for col in df.columns:
        explainers[col] = {}

        for time_start in time_starts:
            time_end = time_start + n_train - 1
            explainers[col][time_end] = {}

            # Recall that correction_model.ar_order=0 for a non auto-regressive model
            correction_windows_actual = [r for r in np.atleast_1d(correction_windows) if r <= time_end - correction_model.ar_order + 1]

            if type(correction_model) == pitsa.models.timeseries.RNN:  # Required for RNN
                correction_model.initialize(lstm_units=16, SEED=2022)
                explainer.correction_model = correction_model

            # Data update
            t = list(df.index)[time_start:time_end+1]
            y = list(df.values.flatten())[time_start:time_end+1]

            for k, correction_window in enumerate(correction_windows_actual):
                print('data: {} - time_step:  {} - correction_window: {}'.format(col, time_end, correction_window), end='\r')
                assert len(y) >= correction_window, 'data at time_start {} must have at least correction_window {} samples'.format(time_start, correction_window)
                
                # Explainer reset, update and forecast
                explainer.reset()
                if k == 0:
                    explainer.update(t=t, y=y, correction_window=correction_window, fit_residual=True, tol=0.01)
                    first_base_model = explainer.first_base_model
                    correction_model = explainer.correction_model
                    first_base_model_at_t = explainer.first_base_model_at_t.copy()
                    correction_model_at_t = explainer.correction_model_at_t.copy()
                else:
                    explainer.first_base_model = first_base_model
                    explainer.correction_model = correction_model
                    explainer.first_base_model_at_t = first_base_model_at_t
                    explainer.correction_model_at_t = correction_model_at_t
                    # fast = True means that just step-3 of the bapc needs to be computed at each new iteration
                    explainer.update(t=t, y=y, correction_window=correction_window, fit_residual=True, fast=True)

                if forecast_lenght > 0:
                    explainer.forecast(forecast_lenght=forecast_lenght)

                # Storing results 
                explainers[col][time_end][correction_window] = explainer.get_components() 
                params_curr = explainer.get_params_as_df().iloc[-1:]
                params_curr['data'] = col
                params_curr['time_start'] = time_start
                params_curr['time_end'] = time_end
                params_curr['correction_window'] = correction_window
                data.append(params_curr)

                k = k + 1

    parameters = pd.concat(data)
    parameters.reset_index(inplace=True, drop=True)

    return parameters, explainers