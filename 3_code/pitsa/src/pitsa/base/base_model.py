import numpy as np
import pandas as pd

def check_timestamps_consistency(t1, t2):
    """
    Check the consistency of timestamps in two arrays.

    This function checks if the timestamps in the input arrays `t1` and `t2` are consistent.
    It performs the following checks:
    - `t1` must be strictly increasing and evenly spaced.
    - `t2` must be strictly increasing and evenly spaced (if it has more than one element).
    - The combined timestamps of `t1` and `t2` (`t1 + t2`) must be strictly increasing and evenly spaced.

    Args:
        t1 (array-like): First array of timestamps.
        t2 (array-like): Second array of timestamps.

    Returns:
        tuple: A tuple containing the start and end indices of `t2` in the combined timestamps (`t1 + t2`).

    Raises:
        AssertionError: If any of the consistency checks fail.

    Example:
        t1 = [0, 1, 2, 3]
        t2 = [4, 5, 6]
        start, end = check_timestamps_consistency(t1, t2)
        print(start)  # Output: 4
        print(end)  # Output: 6
    """
    if isinstance(t1, np.ndarray):
        t1 = list(t1.flatten())

    if isinstance(t2, np.ndarray):
        t2 = list(t2.flatten())

    dt1 = np.diff(t1)[0]
    tol = dt1 / 100

    assert np.all(np.diff(t1) > 0), 'Argument t1 must be strictly increasing'
    assert all(np.abs(np.diff(t1) - dt1) < tol), 'Argument t1 must be evenly spaced'

    t2 = np.atleast_1d(t2)

    if len(t2) > 1:
        dt2 = np.diff(t2)[0]
        assert np.all(np.diff(t2) > 0), 'Argument t2 must be strictly increasing'
        assert all(np.abs(np.diff(t2) - dt2) < tol), 'Argument t2 must be evenly spaced'

    t = t1 + [key for key in t2 if key not in t1]
    dt = np.diff(t)[0]

    assert np.all(np.diff(t) > 0), 't1+t2 must be strictly increasing'
    assert all(np.abs(np.diff(t) - dt) < tol), 't1+t2 must be evenly spaced'

    start = t.index(t2[0])
    end = t.index(t2[-1])

    return start, end


def check_args_1(t, y, as_series=False):
    """
    Validates and transforms input data for consistency in further processing.

    This function checks if the inputs `t` and `y` are in the expected format (either a pandas Series or a pair of index array and data array).
    It then reshapes and/or converts these inputs into a consistent format based on the `as_series` flag.

    Parameters:
    - t (np.ndarray or None): The time/index array. Can be None if `y` is a pandas Series.
    - y (np.ndarray, list, or pd.Series): The data array or a pandas Series containing the data and index.
    - as_series (bool): Flag indicating whether to return `y` as a pandas Series. If False, both `t` and `y` are returned as numpy arrays.

    Returns:
    - tuple: A tuple containing the transformed `t` and `y`.

    Raises:
    - AssertionError: If the input does not meet the expected format.
    """
    is_series = isinstance(y, pd.Series)
    is_array_or_list = isinstance(y, (list, np.ndarray))

    # Validate input formats
    condition1 = is_series and (t is None)
    condition2 = is_array_or_list and (t is not None)
    assert condition1 or condition2, 'Please provide either an input series or an input pair (index, array)'

    if not as_series:
        # Convert inputs to numpy arrays with shape (-1, 1)
        t = np.reshape(y.index.values if condition1 else t, (-1, 1))
        y = np.reshape(y.values if condition1 else y, (-1, 1))
    else:
        # Convert `y` to a pandas Series if not already, with `t` as the index
        if condition2:
            y = pd.Series(np.squeeze(y), index=np.squeeze(t))
        t = np.reshape(y.index.values if condition1 else t, (-1, 1))

    return t, y



def check_args_2(t_ref, t=None, start=None, end=None, as_series=False):
    """
    Validates and prepares time-related arguments for further processing.

    This function ensures that either a range defined by start and end or an array of time-stamps `t` is provided.
    It then calculates or verifies the time-stamps based on the provided arguments and the reference time `t_ref`.
    The function can return the time-stamps as a numpy array or as indices within a pandas Series based on `as_series`.

    Parameters:
    - t_ref (np.ndarray): Reference time array.
    - t (np.ndarray, optional): Array of time-stamps.
    - start (int, optional): Start index for generating time-stamps.
    - end (int, optional): End index for generating time-stamps.
    - as_series (bool): Flag indicating whether to return time-stamps as indices within a pandas Series.

    Returns:
    - tuple: A tuple containing the time-stamps `t` and, if applicable, the start and end indices.

    Raises:
    - AssertionError: If neither a valid range (start, end) nor time-stamps `t` are provided.
    """
    # Validate input conditions
    condition1 = (start is not None) and (end is not None) and (t is None)
    condition2 = (start is None) and (end is None) and (t is not None)
    assert condition1 or condition2, 'Please provide either start, end or time-stamps t'

    if not as_series:
        if condition1:
            dt = np.diff(t_ref)[0]  # Calculate time step based on reference time
            t = t_ref[0] + np.arange(start, end + 1) * dt  # Generate time-stamps within the specified range
    else:
        if condition2:
            start, end = check_timestamps_consistency(t_ref, t)  # Validate and adjust start and end based on `t`

    return t, start, end


class BaseModel():
    """
    Base class for all models.
    """

    def __init__(self, model=None, name='Unnamed-Model', fit_args={}, as_series=False):
        """
        Initialize the BaseModel.

        Args:
            model (object, optional): The model object. Defaults to None.
            name (str, optional): The name of the model. Defaults to 'Unnamed-Model'.
            fit_args (dict, optional): Additional arguments for model fitting. Defaults to {}.
        """
        self.model = model
        self.fit_args = fit_args
        self.as_series = as_series
        self.time_stamps = None
        self.params = None
        #self.ar_order = 0 comment out on 14/11/2024 ... I hope this does not break anything
        self._name = name

    @property
    def name(self):
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._name
    
    @name.setter
    def name(self, val):
        """
        Set the name of the model.

        Args:
            val (str): The name of the model.
        """
        self._name = val

        
    def fit(self,  t=None, y=None, params=None):
        """
        Fit the model to the given data.

        Args:
            t (array-like or None, optional): The array of timestamps. Defaults to None.
            y (array-like or pandas Series, optional): The array of target values. Defaults to None.
            params (dict, optional): The parameters for the model. Defaults to None.

        Returns:
            object: The fitted model object.
        """
        t, y = check_args_1(t,  y, as_series=self.as_series)
        self.time_stamps = t.flatten().tolist()

        if params is None:
            return self._fit(t, y, **self.fit_args)
        else:
            return self.set_params(**params)
    

    def set_params(self, params):
        """
        Set the parameters of the model.

        Args:
            params (dict): The parameters for the model.
        """
        self._set_params(params)

    def get_params(self, **kwargs):
        """
        Get the parameters of the model.

        Returns:
            dict: The parameters of the model.
        """
        return self._get_params(**kwargs)


    def predict(self, t=None, start=None, end=None, **predict_args):
        """
        Predict the target values for the given timestamps.

        Args:
            t (array-like or None, optional): The array of timestamps. Defaults to None.
            start (int or None, optional): The start index. Defaults to None.
            end (int or None, optional): The end index. Defaults to None.
            **predict_args: Additional arguments for prediction.

        Returns:
            array-like: The predicted target values.
        """
        t, start, end = check_args_2(self.time_stamps, t, start, end, as_series=self.as_series)
        y = self._predict(t, start, end, **predict_args)

        return y
    
    
    def forecast(self, forecast_lenght=1, return_time_stamps=False, **predict_args):
        """
        Forecast the target values for the future timestamps.

        Args:
            forecast_length (int, optional): The length of the forecast. Defaults to 1.
            return_time_stamps (bool, optional): Whether to return the forecast timestamps. Defaults to False.
            **predict_args: Additional arguments for forecasting.

        Returns:
            array-like or tuple: The forecasted target values or tuple of timestamps and target values.
        """
        assert forecast_lenght >= 1, 'Assertion: forecast_length must be larger or equal than 1'

        start = len(self.time_stamps)
        end = start + forecast_lenght - 1
        t, start, end = check_args_2(t_ref=self.time_stamps, t=None, start=start, end=end)
        y = self._predict(t, start, end, **predict_args)

        if  return_time_stamps:
            return t, y
        else:
            return y      
