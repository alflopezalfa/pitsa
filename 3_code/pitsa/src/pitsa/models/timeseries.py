import pitsa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from tabulate import tabulate
from IPython.display import display, Math

"""
This module contains classes for timeseries models in the PITSA framework.
"""

class RobustAutoRegressive(pitsa.base.base_model.BaseModel):

    def __init__(self, name='RobustAutoRegressiveRegression', fit_args=None):
        """
        Initialize the RobustAutoRegressive model.

        Args:
            name (str): Name of the model.
            fit_args (dict, optional): Arguments for fitting the model.
        """
        if fit_args is None:
            fit_args = {}

        self.ar_order = 2
        self.robust = fit_args.get('robust', False)
        self.ics = []

        assert self.robust in [False, 'ransac'] or isinstance(self.robust, int), 'robust must be False, "ransac" or an integer'

        if self.robust == 'ransac':
            self.reg = RANSACRegressor(random_state=42)
        else:
            self.reg = LinearRegression(fit_intercept=False)

        # Initialize the parent class with the remaining fit_args
        super().__init__(fit_args=fit_args, name=name, as_series=True)

        self.params = {'ar.L1': np.nan, 'ar.L2': np.nan}  # consistent with statsmodels ARIMA

    def _set_params(self, params):
        """
        Set the parameters of the model.

        Args:
            params (dict): Dictionary of parameters to set in the model.
        """
        self.params.update(params)

    def _get_params(self, format=True):
        """
        Get the parameters of the model.

        Returns:
            dict: Dictionary of model parameters.
        """
        return self.params
    

    def _fit(self, t=None, y=None, **kwargs):
        """
        Fit the model to the provided time series data.

        Args:
            t (optional): Time indices for the time series data. Not used in this method.
            y (array-like): Time series data to fit the model to.
            **kwargs: Additional keyword arguments for fitting the model.

        Returns:
            None
        """
        X, Y = self.timeseries_dataset_from_array(np.reshape(y, (-1, 1)), ar_order=self.ar_order)

        if isinstance(self.robust, int):
            # remove index of outliers at self.robust,..,self.robust+self.ar_order
            X = np.delete(X, np.arange(self.robust - self.ar_order, self.robust), axis=0)
            Y = np.delete(Y, np.arange(self.robust - self.ar_order, self.robust), axis=0)

        self.reg.fit(X, Y)

        if self.robust == 'ransac':
            self.params['ar.L1'] = self.reg.estimator_.coef_[0][1]
            self.params['ar.L2'] = self.reg.estimator_.coef_[0][0]
            inlier_mask = self.reg.inlier_mask_
            outlier_mask = ~inlier_mask
            print(f"Total outliers: {sum(outlier_mask)}")
        else:
            self.params['ar.L1'] = self.reg.coef_[0][1]
            self.params['ar.L2'] = self.reg.coef_[0][0]

        self.ics = list(y[:self.ar_order])  # Recall that y could be an array or a series

    def _predict(self, t, start, end, **predict_args):
        """
        Predict future values using the fitted model.

        Args:
            t: Time indices for the time series data. Not used in this method.
            start (int): The starting index for prediction.
            end (int): The ending index for prediction.
            **predict_args: Additional keyword arguments for the prediction method.

        Returns:
            np.ndarray: Predicted values reshaped to a column vector.
        """
        dynamic = predict_args.get('dynamic', True)
        assert dynamic, 'RobustAutoRegressive model does not support non-dynamic prediction'

        y = self.ics.copy()

        for i in range(self.ar_order, end + 1):
            yi = np.array(y[-self.ar_order:]).reshape(1, -1)
            y.append(self.reg.predict(yi)[0][0])

        return np.array(y[start:]).reshape(-1, 1)
    
    
    def simulate(self, start, end):

        phi1 = self.params['ar.L1']
        phi2 = self.params['ar.L2']

        y = self.ics.copy()

        for i in range(self.ar_order, end + 1):
            y.append(phi1*y[i-1] + phi2*y[i-2])

        return np.array(y[start:]).reshape(-1, 1)

    @staticmethod
    def timeseries_dataset_from_array(data, ar_order):
        """
        Create a dataset matrix from a time series array.

        Args:
            data (array-like): Time series data.
            ar_order (int): Autoregressive order.

        Returns:
            tuple: A tuple containing the input features (X) and target values (y).
        """
        X, y = [], []
        for i in range(len(data) - ar_order):
            a = data[i:(i + ar_order), 0]
            X.append(a)
            y.append(data[i + ar_order, 0])
        return np.array(X), np.array(y).reshape(-1, 1)


class RNN(pitsa.base_model.BaseModel):

    def __init__(self, model, ar_order, name='RNN', fit_args={}):

        super().__init__(model=model, name=name, fit_args=fit_args, as_series=True)
        self.ar_order = ar_order
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = None
        self.Xt = None
        self.yt = None
        self.fit_flag = False


    def initialize(self, lstm_units, SEED):
        
        model = sequential_lstm(lstm_units=lstm_units, N_STEPS_IN=self.ar_order, SEED=SEED)
        model.compile(loss='mean_squared_error', optimizer='adam') 
        self.model = model


    # convert an array of values into a dataset matrix
    def timeseries_dataset_from_array(self, data, ar_order=1):
        X, y = [], []
        for i in range(len(data)-ar_order):#-1):
            a = data[i:(i+ar_order), 0]
            X.append(a)
            y.append(data[i + ar_order, 0])
        return np.array(X), np.array(y)
    

    def _fit(self, t, y, **kwargs):

        # assumes t is None and y is a time-series

        # normalize the dataset
        self.scaled_data = self.scaler.fit_transform(np.reshape(y.values, (-1, 1)))
        # reshape into X=t and Y=t+1
        Xt, yt = self.timeseries_dataset_from_array(self.scaled_data, self.ar_order)
        # reshape input to be [samples, time steps, features]
        Xt = np.reshape(Xt, (Xt.shape[0], 1, Xt.shape[1]))
        self.Xt = Xt
        self.yt = yt
        # fix random seed for reproducibility
        tf.random.set_seed(7)
        # fit the LSTM network
        self.model.fit(Xt, yt, **kwargs) 
        self.fit_flag = True


    def predict_insample(self, data, start, end, inverse_transform=True):
            
            assert start >= self.ar_order, 'start must be larger or equal than ar_order'
            assert end <= len(data), 'end must be lower or equal than the number of samples'
            assert len(data)>0, 'input data is empty'

            # subset and augment data: we don't include data[end] as it is not used for prediction and might not even 
            # be available, therefore we augment data with a dummy value
            right = np.expand_dims(np.full_like(data[0], np.nan, dtype=float), 0)
            data = np.concatenate((data[start-self.ar_order:end], right)) 
            # reshape into X=t and Y=t+1
            X, _ = self.timeseries_dataset_from_array(data, self.ar_order)
            # reshape input to be [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            # make predictions
            y = self.model.predict(X, verbose=0)
            # invert predictions
            if inverse_transform:
                y = self.scaler.inverse_transform(y)
            
            return y
    

    def predict_dynamic(self, data, start, end, inverse_transform=True):

        assert start<=end, 'start must be lower or equal than end'
        assert start >= self.ar_order, 'start must be larger or equal than ar_order'
        assert len(data)>0, 'input data is empty'

        start_prime = np.minimum(start, len(data))
        y = data[0:start_prime]

        for t in range(start_prime, end+1):
            yt = self.predict_insample(data=y, start=t, end=t, inverse_transform=False)
            y = np.concatenate((y, yt))

        y = y[start:]

        # invert predictions
        if inverse_transform:
            y = self.scaler.inverse_transform(y)
        
        return y
    
    def _predict(self, t, start, end, **predict_args):

        dynamic = predict_args.pop('dynamic', False)

        if not dynamic:
            dynamic = False

        start, end, left, right = check_args_3(start=start,
                                               end=end, 
                                               n_samples=len(self.scaled_data), 
                                               ar_order=self.ar_order,
                                               dynamic=dynamic,
                                               strong_assertion=False)


        if dynamic:
            y = self.predict_dynamic(self.scaled_data, start, end)
        else:
            y = self.predict_insample(self.scaled_data, start, end)
            # comment out the following 2 lines for a strict non-dynamic behaviour
            if len(right)>0:
                right = self.forecast(forecast_lenght=len(right), dynamic=True)

        y = np.concatenate((left, y, right))

        return y
    
    def predict_from_array(self, X, reescale=True):
        """
        Predict future values using the fitted model.

        Args:
            X (np.ndarray): Array of input features.

        Returns:
            np.ndarray: Predicted values reshaped to a column vector.
        """
        # Check if the model is fitted
        assert self.model is not None, 'Model is not fitted'
        assert X.shape[1] == self.ar_order, 'Input features must have the same number of columns as the AR order'

        # Normalize the input features
        if reescale:
            
            a = []
            for x in X:
                a.append(self.scaler.transform(np.reshape(x, (-1, 1))))
            X = np.array(a)

        # Reshape the input features
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # Predict the output values
        y = self.model.predict(X, verbose=0)

        # Inverse transform the output values
        if reescale:
            y = self.scaler.inverse_transform(y)

        return y
    

def sequential_lstm(lstm_units, N_STEPS_IN, SEED): 

    N_STEPS_OUT = 1  # set N_STEPS_OUT >1 for multi-step

    # Use initializers to initialize model parameters with the same values
    INITIALIZER_GLOROT_UNIFORM = tf.keras.initializers.GlorotUniform(seed = SEED)
    INITIALIZER_ORTHOGONAL = tf.keras.initializers.Orthogonal(gain = 1.0, seed = SEED)

    model = Sequential()
    model.add(Input(shape=(1, N_STEPS_IN)))
    model.add(LSTM(units                 = lstm_units, 
                   activation            = 'relu',
                   #input_shape           = (1, N_STEPS_IN), 
                   kernel_initializer    = INITIALIZER_GLOROT_UNIFORM, 
                   recurrent_initializer = INITIALIZER_ORTHOGONAL,
                   recurrent_dropout     = 0.4))
    model.add(Dense(units              = N_STEPS_OUT,
                    kernel_initializer = INITIALIZER_GLOROT_UNIFORM))
    return model


def stacked_lstm(lstm_units, N_STEPS_IN, SEED):

    N_STEPS_OUT = 1  # set N_STEPS_OUT >1 for multi-step

    # Use initializers to initialize model parameters with the same values
    INITIALIZER_GLOROT_UNIFORM = tf.keras.initializers.GlorotUniform(seed = SEED)
    INITIALIZER_ORTHOGONAL = tf.keras.initializers.Orthogonal(gain = 1.0, seed = SEED)

    model = Sequential()
    model.add(Input(shape=(1, N_STEPS_IN)))
    model.add(LSTM(units                 = lstm_units, 
                   activation            = 'relu',
                   #input_shape           = (1, N_STEPS_IN), 
                   kernel_initializer    = INITIALIZER_GLOROT_UNIFORM, 
                   recurrent_initializer = INITIALIZER_ORTHOGONAL,
                   recurrent_dropout     = 0.4,
    ##########################################################################
    # NEWLY ADDED
    # https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
    ##########################################################################
                   return_sequences      = True)) 
    model.add(LSTM(units                 = lstm_units, 
                   kernel_initializer    = INITIALIZER_GLOROT_UNIFORM, 
                   recurrent_initializer = INITIALIZER_ORTHOGONAL))
    ##########################################################################
    
    model.add(Dense(units              = N_STEPS_OUT,
                    kernel_initializer = INITIALIZER_GLOROT_UNIFORM))
    return model
    

def check_args_3(start, end, ar_order, n_samples, dynamic, strong_assertion=False):
    """
    Check the arguments for consistency and reshape them if necessary.

    This function checks the input arguments `start`, `end`, `ar_order`, `n_samples`, `dynamic`, and `strong_assertion` for consistency and reshapes them if necessary.
    It performs the following checks:
    - `start` must be lower or equal to `end`.
    - If `strong_assertion` is True, `start` must be larger or equal to `ar_order`.
    - If `dynamic` is False, `end` must be lower or equal to `n_samples`.
    - If `strong_assertion` is False, `start` must be larger or equal to zero.
    - If `strong_assertion` is False and `ar_order` is larger than `start`, fill `left` with NaN values and update `start`.
    - If `strong_assertion` is False and `dynamic` is False and `n_samples` is smaller than `end`, fill `right` with NaN values and update `end`.

    Args:
        start (int): Start index.
        end (int): End index.
        ar_order (int): Autoregressive order.
        n_samples (int): Number of samples.
        dynamic (bool): Flag indicating if the model is dynamic.
        strong_assertion (bool, optional): Flag indicating if strong assertions should be applied. Defaults to False.

    Returns:
        tuple: A tuple containing the updated `start`, `end`, `left`, and `right` values.

    Raises:
        AssertionError: If any of the consistency checks fail.

    Example:
        start, end, left, right = check_args_3(start=2, end=6, ar_order=1, n_samples=10, dynamic=True)
        print(start)  # Output: 2
        print(end)  # Output: 6
        print(left)  # Output: array([], shape=(0, 1), dtype=float64)
        print(right)  # Output: array([], shape=(0, 1), dtype=float64)
    """
    # Check conditions
    assert start <= end, 'start must be lower or equal than end'

    # Initialize left and right arrays
    left = np.array([]).reshape(-1, 1)
    right = np.array([]).reshape(-1, 1)

    if strong_assertion:
        assert start >= ar_order, 'start must be larger or equal than ar_order'
        if not dynamic:
            assert end <= n_samples, 'end must be lower or equal than the number of samples'
    else:
        assert start >= 0, 'start must be larger or equal than zero'
        if not start >= ar_order:
            left = np.full((ar_order, 1), np.nan, dtype=float)
            start = ar_order
        if not dynamic:
            if not end <= n_samples:
                right = np.full((end - n_samples, 1), np.nan, dtype=float)
                end = n_samples

    return start, end, left, right