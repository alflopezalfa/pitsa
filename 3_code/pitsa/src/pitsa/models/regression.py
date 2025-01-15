import pitsa

import numpy as np
import sklearn

import sklearn.pipeline
import sklearn.neighbors
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from functools import partial

import numpy as np
import chaospy

"""
This module contains classes for regression models in the PITSA framework.
"""


class Sinusoidal(pitsa.base_model.BaseModel):
    """
    Sinusoidal model class that combines polynomial and sinusoidal components.
    """

    def __init__(self, name='Sinusoidal', fit_args={}):
        """
        Initialize the Sinusoidal model.

        Parameters:
        name (str, optional): Name of the model. Defaults to 'Sinusoidal'.
        fit_args (dict, optional): Additional arguments for fitting the model. Defaults to {}.
        """
        super().__init__(name=name, fit_args=fit_args)
        self.params = {'intercept': np.nan, 
                       'slope' : np.nan, 
                       'quadratic':np.nan, 
                       'amplitude': np.nan, 
                       'frequency': np.nan, 
                       'phase': np.nan}
        
    def _set_params(self, params):
        """
        Set the parameters of the model.

        Parameters:
        params (dict): Dictionary of parameters to set.
        """
        self.params.update(params)

    def _get_params(self, format=False):
        """
        Get the parameters of the model.

        Parameters:
        format (bool, optional): Whether to format the parameters. Defaults to False.

        Returns:
        dict: Dictionary of model parameters.
        """
        return self.params
    
    def _fit_poly(self, t, y):
        """
        Fit the polynomial component of the model.

        Parameters:
        t (array): Time steps.
        y (array): Observations.

        Returns:
        array: Residuals after fitting the polynomial component.
        """
        t = np.array(t).flatten()
        y = np.array(y).flatten()
        q0 = chaospy.variable()
        terms = np.power(q0, [0, 1, 2]) # hard coded for now
        expansion = chaospy.polynomial(terms)
        poly = chaospy.fit_regression(expansion, t, y)
        coef = poly.coefficients
        resid = y - poly(t)
       
        self.params['intercept'] = coef[0]
        self.params['slope'] = coef[1]
        self.params['quadratic'] = coef[2]

        return resid

    def _fit_cos(self, t, y, params_init, params_fixed, **kwargs):
        """
        Fit the sinusoidal component of the model.

        Parameters:
        t (array): Time steps.
        y (array): Observations.
        params_init (dict): Initial parameters for fitting.
        params_fixed (dict): Fixed parameters for fitting.
        **kwargs: Additional arguments for fitting.
        """
        cos  = Cosine(fit_args={'params_init': params_init, 'params_fixed': params_fixed})
        cos.fit(t, y)

        self.params['amplitude'] = cos.params['amplitude']
        self.params['frequency'] = cos.params['frequency']
        self.params['phase'] = cos.params['phase']

    def _fit(self, t, y, params_init, params_fixed, **kwargs):
        """
        Fit the model by combining polynomial and sinusoidal components.

        Parameters:
        t (array): Time steps.
        y (array): Observations.
        params_init (dict): Initial parameters for fitting.
        params_fixed (dict): Fixed parameters for fitting.
        **kwargs: Additional arguments for fitting.
        """
        resid = self._fit_poly(t, y)
        self._fit_cos(t, resid, params_init, params_fixed, **kwargs)

    def _predict(self, t, start, end):
        """
        Predict the model output for the given time steps.

        Parameters:
        t (array): Time steps.
        start (int): Start index for prediction.
        end (int): End index for prediction.

        Returns:
        array: Predicted values.
        """
        y = self.params['intercept'] + self.params['slope']*t + self.params['quadratic']*t**2 + self.params['amplitude']*np.cos(self.params['frequency']*t + self.params['phase'])
        return y.reshape(-1, 1)

class Cosine(pitsa.base_model.BaseModel):
    """
    A class representing the Cosine model.

    This model represents a sinusoidal function of the form:
    y = intercept + amplitude * cos(frequency * x + phase)

    Parameters:
    - name (str): The name of the model (default: 'Cosine')
    - fit_args (dict): Additional arguments for fitting the model (default: {})

    Attributes:
    - params (dict): The parameters of the model, including 'intercept', 'amplitude', 'frequency', and 'phase'

    Methods:
    - _sinusoidal(x, intercept, amplitude, frequency, phase): Computes the sinusoidal function
    - _sinusoidal_jac(x, intercept, amplitude, frequency, phase): Computes the Jacobian of the sinusoidal function
    - _fit_amplitude(t, y, params_init, **kwargs): Fits the amplitude parameter of the model
    - _fit(t, y, params_init, params_fixed, **kwargs): Fits the model parameters
    - _predict(t, start, end): Predicts the output of the model for the given time range
    """
    def __init__(self, name='Cosine', fit_args={}):
        # Initialize the base model with the given name and fit arguments
        super().__init__(name=name, fit_args=fit_args)
        # Initialize model parameters with NaN values
        self.params = {'intercept': np.nan, 'amplitude': np.nan, 'frequency': np.nan, 'phase': np.nan}


    def _set_params(self, params):
        """
        Set parameters for the model and update related attributes.

        Args:
            params (dict): Dictionary containing model parameters.
        """
        self.params.update(params)

    def _get_params(self, format=False):
        """
        Get parameters for the model.

        Args:
            format (bool): Whether to format the parameters based on the model type.

        Returns:
            dict: Dictionary containing model parameters.
        """
        return self.params

    def _sinusoidal(self, x, intercept, amplitude, frequency, phase):
        # Compute the sinusoidal function based on the given parameters
        return intercept + amplitude * np.cos(frequency * x + phase)

    def _sinusoidal_jac(self, x, intercept, amplitude, frequency, phase):
        # Compute the Jacobian matrix of the sinusoidal function with respect to each parameter
        jac_intercept = amplitude * np.cos(frequency * x + phase)
        jac_amplitude = np.cos(frequency * x + phase)
        jac_frequency = -amplitude * np.sin(frequency * x + phase) * x
        jac_phase = -amplitude * np.sin(frequency * x + phase)
        return np.array([jac_intercept, jac_amplitude, jac_frequency, jac_phase])

    def _fit_amplitude(self, t, y, params_init, **kwargs):
        """
        Fits the amplitude parameter of the model.
        Parameters:
        - t (array-like): The input time values
        - y (array-like): The output values
        - params_init (dict): The initial values for the model parameters
        - **kwargs: Additional arguments for fitting the model
        Returns:
        None
        """
        # Extract initial parameters
        intercept = params_init['intercept']
        amplitude = params_init['amplitude']
        frequency = params_init['frequency']
        phase = params_init['phase']
        # Compute the sinusoidal function based on initial parameters
        X = self._sinusoidal(t, intercept, amplitude, frequency, phase)
        # Fit the amplitude parameter using linear regression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X.reshape(-1, 1), y, **kwargs)
        # Update the model parameters with the fitted values
        self.params['intercept'] = intercept
        self.params['amplitude'] = reg.coef_[0]
        self.params['frequency'] = frequency
        self.params['phase'] = phase

    def _fit_intercept_and_amplitude(self, t, y, params_init, **kwargs):
        """
        Fits the intercept and amplitude parameters of the model.
        Parameters:
        - t (array-like): The input time values
        - y (array-like): The output values
        - params_init (dict): The initial values for the model parameters
        - **kwargs: Additional arguments for fitting the model
        Returns:
        None
        """
        # Extract initial parameters
        intercept = params_init['intercept']
        amplitude = params_init['amplitude']
        frequency = params_init['frequency']
        phase = params_init['phase']
        # Compute the sinusoidal function based on initial parameters
        X = self._sinusoidal(t, intercept, amplitude, frequency, phase)
        # Fit the intercept and amplitude parameters using linear regression
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X.reshape(-1, 1), y, **kwargs)
        # Update the model parameters with the fitted values
        self.params['intercept'] = reg.intercept_
        self.params['amplitude'] = reg.coef_[0]
        self.params['frequency'] = frequency
        self.params['phase'] = phase

    def _fit(self, t, y, params_init, params_fixed, **kwargs):
        """
        Fit the model to the data using a sinusoidal function with dynamic parameters.

        This method fits the model to the data by adjusting the parameters of a sinusoidal function to minimize the difference between the observed data and the model's predictions. It supports fitting with various combinations of fixed and dynamic parameters.

        Parameters:
            t (array-like): The time points at which the data y was measured. Must be a 1D array.
            y (array-like): The measured data corresponding to each time point in t. Must be a 1D array.
            params_init (dict): Initial guesses for the dynamic parameters to be optimized. The keys should be the names of the parameters ('amplitude', 'frequency', 'phase'), and the values should be the initial guesses.
            params_fixed (dict): Parameters that should be kept fixed during the fitting process. The keys should be the names of the parameters to be fixed, and the values should be their fixed values.
            **kwargs: Additional keyword arguments to be passed to the curve fitting function.

        The method first checks that each parameter is declared as either initial or fixed, but not both. It then constructs partial functions for the sinusoidal model and its Jacobian, fixing the values of any fixed parameters. Depending on which parameters are dynamic (i.e., to be fitted), it sets up the appropriate model function `f` and Jacobian `jac` for curve fitting.

        The curve fitting is performed using `curve_fit` from `scipy.optimize`, which requires the model function, its Jacobian, the data to fit, and initial guesses for the dynamic parameters. After fitting, the optimized parameter values are stored in the model's `params` attribute.

        Raises:
            AssertionError: If a parameter is incorrectly declared as both initial and fixed, or if an unsupported combination of fixed parameters is provided.

        Note:
            This method supports fitting models with any combination of 'amplitude', 'frequency', and 'phase' as dynamic parameters. If a combination of fixed parameters is not supported, an assertion error is raised.
        """

        # Ensure each parameter is either fixed or dynamic, but not both
        for key in self.params:
            assert (key in params_init) ^ (key in params_fixed), 'Error: parameter {} must be declared as initial XOR dynamic'.format(key)

        # Construct partial functions for the sinusoidal model and its Jacobian
        f_partial = partial(self._sinusoidal, **params_fixed)
        jac_partial = partial(self._sinusoidal_jac, **params_fixed)
        params_dynamic = list(params_init.keys())

        # Set up the model function and Jacobian based on the dynamic parameters
        if params_dynamic == ['intecrept']:
            f = lambda x, i : f_partial(x, intercept=i)
            jac = lambda x, i : jac_partial(x, intercept=i)[[0]]
        if  params_dynamic == ['amplitude']:
            #self._fit_amplitude(t, y, params_init={**params_init, **params_fixed})
            f = lambda x, a : f_partial(x, amplitude=a)
            jac = lambda x, a : jac_partial(x, amplitude=a)[[1]]
        elif params_dynamic == ['frequency']:
            f = lambda x, f : f_partial(x, frequency=f)
            jac = lambda x, f : jac_partial(x, frequency=f)[[2]]
        elif params_dynamic == ['phase']:
            f = lambda x, p : f_partial(x, phase=p)
            jac = lambda x, p : jac_partial(x, phase=p)[[3]]
        elif params_dynamic == ['intercept', 'amplitude']:
            f = lambda x, i, a : f_partial(x, intercept=i, amplitude=a)
            jac = lambda x, i, a : np.transpose(jac_partial(x, intercept=i, amplitude=a)[[0, 1]])
        elif params_dynamic == ['amplitude','frequency']:
            f = lambda x, a, f : f_partial(x, amplitude=a, frequency=f)
            jac = lambda x, a, f : np.transpose(jac_partial(x, amplitude=a, frequency=f)[[1, 2]])
        elif params_dynamic == ['amplitude','phase']:
            f = lambda x, a, p : f_partial(x, amplitude=a, phase=p)
            jac = lambda x, a, p : np.transpose(jac_partial(x, amplitude=a, phase=p)[[1, 3]])
        elif params_dynamic == ['frequency','phase']:
            f = lambda x, f, p : f_partial(x, frequency=f, phase=p)
            jac = lambda x, f, p : np.transpose(jac_partial(x, frequency=f, phase=p)[[2, 3]])
        elif params_dynamic == ['intercept','amplitude','phase']:
            f = lambda x, i, a, p : f_partial(x, intercept=i, amplitude=a, phase=p)
            jac = lambda x, i, a, p : np.transpose(jac_partial(x, intercept=i, amplitude=a, phase=p)[[0, 1, 3]])
        elif params_dynamic == ['amplitude','frequency','phase']:
            f = lambda x, a, f, p : f_partial(x, amplitude=a, frequency=f, phase=p)
            jac = lambda x, a, f, p : np.transpose(jac_partial(x, amplitude=a, frequency=f, phase=p)[[1, 2, 3]])
        else:
            flag=False
            assert flag, 'The choosen fixed parameters configuration {} is not yet being implemented'.format(params_fixed)


     
        # Extract initial parameter values
        p0 = list(params_init.values())
        # Perform curve fitting to optimize the dynamic parameters
        popt, pcov = curve_fit(f=f, 
                            jac=jac,
                            xdata=np.ravel(t), 
                            ydata=np.ravel(y), 
                            p0=p0, 
                             **kwargs)
            
        # Update the model parameters with the optimized values
        for key in params_fixed:
            self.params[key] = params_fixed[key]

        # Update the model parameters with the optimized values
        for k, key in enumerate(params_dynamic):
            self.params[key] = popt[k]
        # Extract initial parameter values
        p0 = list(params_init.values())
        # Perform curve fitting to optimize the dynamic parameters
        popt, pcov = curve_fit(f=f, 
                            jac=jac,
                            xdata=np.ravel(t), 
                            ydata=np.ravel(y), 
                            p0=p0, 
                            **kwargs)
            
        # Update the model parameters with the optimized values
        for key in params_fixed:
            self.params[key] = params_fixed[key]

        # Update the model parameters with the optimized values
        for k, key in enumerate(params_dynamic):
            self.params[key] = popt[k]


    def _predict(self, t, start, end):
        """
        Predicts the output values for the given time range.

        Args:
            t (array-like): The time values.
            start (int): The start index of the time range.
            end (int): The end index of the time range.

        Returns:
            array-like: The predicted output values.

        """
        y = self._sinusoidal(t, **self.params)
        #y = y.reshape(-1, 1) #FLAG: I commented this out ... I hope does not bring troubles
        return y
    

class SKLearnModel(pitsa.base_model.BaseModel):
    """
    Wrapper class for scikit-learn models in PITSA.

    Args:
        model: The scikit-learn model object.
        name (str): The name of the model (default: 'Sklearn-Model').
        fit_args (dict): Additional arguments to be passed to the `fit` method of the model (default: {}).
    """

    def __init__(self, model, name='Sklearn-Model', fit_args={}):
        super().__init__(model=model, name=name, fit_args=fit_args)
        self.params = self._initialize_params()

    def _initialize_params(self):
        """
        Initializes model parameters based on the type of the scikit-learn model.

        Returns:
            dict: Initialized parameters.
        """
        # Using a mapping of model classes to their parameter initialization logic
        param_initializers = {
            sklearn.neighbors.KNeighborsRegressor : lambda: {'n_neighbors': np.nan},
            sklearn.dummy.DummyRegressor: lambda: {'intercept': np.nan},
            sklearn.linear_model.LinearRegression: lambda: {'intercept': np.nan, 'slope': np.nan},
            sklearn.pipeline.Pipeline: lambda: {'intercept': np.nan, **{f'coef_{k+1}': np.nan for k in range(self.model['polynomial_features'].degree)}}
        }
        
        for model_class, initializer in param_initializers.items():
            if isinstance(self.model, model_class):
                return initializer()
        return {}
    
    def _update_params(self) : 
        """
        Updates model parameters based on the fitted model.
        """
        # Using a mapping of model classes to their parameter initialization logic
        param_initializers = {
            sklearn.neighbors.KNeighborsRegressor : lambda: {'n_neighbors': self.model.n_neighbors},
            sklearn.dummy.DummyRegressor: lambda: {'intercept': self.model.constant_[0][0]},
            sklearn.linear_model.LinearRegression: lambda: {'intercept':self.model.intercept_[0], 'slope': self.model.coef_[0][0]},
            sklearn.pipeline.Pipeline: lambda: {'intercept':self.model['linear_regression'].intercept_[0], 
                                                **{f'coef_{k+1}': self.model['linear_regression'].coef_[0] for k in range(self.model['polynomial_features'].degree)}}
        }

        params = False

        for model_class, initializer in param_initializers.items():
            if isinstance(self.model, model_class):
                params = initializer()

        assert params, 'The model class {} is not yet being implemented'.format(self.model.__class__)
        
        for key in params:
            self.params[key] = params[key]            


    def _set_params(self, params):
        """
        Set parameters for the model.

        Args:
            params (dict): Dictionary containing model parameters.
        """
        if isinstance(self.model, sklearn.dummy.DummyRegressor):
            self.model.constant_[0][0] = params['intercept']
        elif isinstance(self.model, sklearn.linear_model.LinearRegression):
            self.model.intercept_[0] = params['intercept']
            self.model.coef_[0][0] = params['slope']
        elif isinstance(self.model, sklearn.pipeline.Pipeline):
            self.model['linear_regression'].intercept_[0] = params['intercept']
            for k in range(self.model['polynomial_features'].degree):
                self.model['linear_regression'].coef_[0][k] = params[f'coef_{k+1}']

    def _get_params(self, format=False):
        """
        Get parameters for the model.

        Returns:
            dict: Dictionary containing model parameters.
        """
        return self.params
        

    def _fit(self, t, y, **kwargs):
        """
        Fits the model to the given training data.

        Args:
            t (array-like): The input training data.
            y (array-like): The target training data.
            **kwargs: Additional keyword arguments to be passed to the `fit` method of the model.
        """
        self.model.fit(t, y, **kwargs)
        self._update_params()

    def _predict(self, t, start, end):
        """
        Predicts the target values for the given input data.

        Args:
            t (array-like): The input data.
            start: The start index of the prediction range.
            end: The end index of the prediction range.

        Returns:
            array-like: The predicted target values.
        """
        t = np.reshape(t, (-1,1))  # FLAG: I hope that one day I can get rid-off too many reshape
        y = self.model.predict(t)
        return y
    


    
    