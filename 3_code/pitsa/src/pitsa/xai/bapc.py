import warnings
import numpy as np
import pitsa as pitsa
import pitsa.utils.common as ut
import pitsa.xai.neighbors as nbh 
from copy import deepcopy


class Explainer(object):
    """
    Explainer class.
    """

    def __init__(self, base_model, correction_model, predict_args={}):
        """
        Initialize the Explainer with base models and correction model.

        Parameters:
        base_model (object): The base model, this is f_{\theta} in the paper
        correction_model (object): The correction model, this is \widehat{\varepsilon} in the paper.
        predict_args (dict, optional): Additional arguments for the predict method. Defaults to {}.
        """
        
        self.first_base_model = deepcopy(base_model)   # this is f_{\theta_0} in the paper
        self.second_base_model = deepcopy(base_model)  # this is f_{\theta_r} in the paper
        self.correction_model = correction_model
        self.predict_args = predict_args
        self.reset()

    def reset(self):
        """
        Reset the state of the Explainer, including models and internal variables.
        """

        # Reset base models if they have a reset method
        if hasattr(self.first_base_model, 'reset'):
            self.first_base_model.reset()
        if hasattr(self.second_base_model, 'reset'):
            self.second_base_model.reset()

        # Initialize lists to store time steps and model predictions
        self.t = []
        self.y = []
        self.first_base_model_at_t = []    
        self.correction_model_at_t = []  
        self.second_base_model_at_t  = []  
        self.full_model_at_t = []
        self.surrogate_model_at_t = []

        # Initialize lists to store test time steps and model predictions
        self.t_test = []
        self.first_base_model_at_t_test = []  
        self.correction_model_at_t_test = []    
        self.second_base_model_at_t_test  = []  
        self.full_model_at_t_test = []
        self.surrogate_model_at_t_test = []

        # Initialize other internal variables
        self.residual = None              
        self.neighborhood = None           
        self.y_modified = None
        self.fit_residual = None

        # Initialize parameter dictionaries for base models
        params = {key: [] for key in self.first_base_model.params} 
        self.first_params_at_t = deepcopy(params) # important to make a deep copy!
        self.second_params_at_t = deepcopy(params) # important to make a deep copy!
        self.delta_params_at_t = deepcopy(params) # important to make a deep copy!
       


    def fit(self, model, model_at_t, tprev, yprev, t=None, y=None, predict_args={}, params=None):
        """
        Fit the model with the provided data and update the model_at_t.
    
        Parameters:
        model (object): The model to be fitted.
        model_at_t (list): The list to store model predictions.
        tprev (list): Previous time steps.
        yprev (list): Previous observations.
        t (list, optional): New time steps. Defaults to None.
        y (list, optional): New observations. Defaults to None.
        predict_args (dict, optional): Additional arguments for the model's predict method. Defaults to {}.
        params (dict, optional): Parameters to set for the model. Defaults to None.
        """
        
        # Append new time steps and observations if provided
        if (t is not None) and (y is not None):
            for tc, yc in zip(np.atleast_1d(t), np.atleast_1d(y)):
                tprev.append(tc)
                yprev.append(yc)
    
        # Reshape time and observation data for training
        t_train = np.reshape(tprev, (-1,1))
        y_train = np.reshape(yprev, (-1,1))
    
        # Set model parameters if provided
        if params is not None:
            model.set_params(params)
        else:
            model.fit(t_train, y_train)
    
        # Predict using the model
        y_pred = model.predict(t_train, **predict_args)
        
        # Update model_at_t with the predictions
        model_at_t[:] = y_pred.flatten()


    def update1(self, t=None, y=None):
        """
        BAPC Step-1.
    
        Parameters:
        t (list, optional): New time steps. Defaults to None.
        y (list, optional): New observations. Defaults to None.
        """
        
        # Fit the first base model with the provided data
        self.fit(
            model=self.first_base_model, 
            model_at_t=self.first_base_model_at_t, 
            tprev=self.t, 
            yprev=self.y, 
            t=t, 
            y=y, 
            predict_args=self.predict_args
        )
        
        # Append NaN values to the first_params_at_t dictionary for all but the last new time step
        for k in range(len(np.atleast_1d(t))-1):
            ut.append_dict(self.first_params_at_t, np.nan)
    
        # Append the current parameters of the first base model to the first_params_at_t dictionary
        ut.append_dict(self.first_params_at_t, self.first_base_model.params)
     


    def update1_fast(self, t=None, y=None):
        """
        Quickly update the first base model with new data without fitting the model.
    
        Parameters:
        t (list, optional): New time steps. Defaults to None.
        y (list, optional): New observations. Defaults to None.
        """
        
        # Append new time steps and observations if provided
        if (t is not None) and (y is not None):
            for tc, yc in zip(np.atleast_1d(t), np.atleast_1d(y)):
                self.t.append(tc)
                self.y.append(yc)
        
        # Append NaN values to the first_params_at_t dictionary for all but the last new time step
        for k in range(len(np.atleast_1d(t))-1):
            ut.append_dict(self.first_params_at_t, np.nan)
    
        # Append the current parameters of the first base model to the first_params_at_t dictionary
        ut.append_dict(self.first_params_at_t, self.first_base_model.params)

     

    def update2(self, fit_residual=True, tol=0):
        """
        BAPC Step-2.
    
        Parameters:
        fit_residual (bool, optional): Whether to fit the residuals. Defaults to True.
        tol (float, optional): Tolerance for residual fitting. Defaults to 0.
        """
        
        # Calculate the residuals between the observations and the first base model predictions
        self.residual = np.reshape(self.y, (-1, 1)) - np.reshape(self.first_base_model_at_t, (-1, 1))
    
        if fit_residual:
            # Check if the maximum absolute deviation of the residuals is within the tolerance
            if np.max(np.abs(self.residual - np.mean(self.residual))) <= tol:
                # If within tolerance, set the correction model predictions to the mean residual
                self.correction_model_at_t = len(self.t) * [np.mean(self.residual)]
            else:
                # Otherwise, fit the correction model with the residuals
                self.fit(
                    model=self.correction_model, 
                    model_at_t=self.correction_model_at_t, 
                    tprev=self.t, 
                    yprev=self.residual
                )
            # Update the full model predictions by adding the correction model predictions to the first base model predictions
            self.full_model_at_t = [y1 + y2 for y1, y2 in zip(self.first_base_model_at_t, self.correction_model_at_t)]
        else:
            # Fit the correction model with the original observations
            self.fit(
                model=self.correction_model, 
                model_at_t=self.full_model_at_t, 
                tprev=self.t, 
                yprev=self.y
            )
            # Update the correction model predictions by subtracting the first base model predictions from the full model predictions
            self.correction_model_at_t = [y1 - y2 for y1, y2 in zip(self.full_model_at_t, self.first_base_model_at_t)]
    

    def update2_fast(self, fit_residual=True):
        """
        Quickly update the correction model and full model based on the residuals without fitting.
    
        Parameters:
        fit_residual (bool, optional): Whether to fit the residuals. Defaults to True.
        """
        
        # Calculate the residuals between the observations and the first base model predictions
        self.residual = np.reshape(self.y, (-1, 1)) - np.reshape(self.first_base_model_at_t, (-1, 1))
    
        if fit_residual:
            # Update the full model predictions by adding the correction model predictions to the first base model predictions
            self.full_model_at_t = [y1 + y2 for y1, y2 in zip(self.first_base_model_at_t, self.correction_model_at_t)]
        else:
            # Update the correction model predictions by subtracting the first base model predictions from the full model predictions
            self.correction_model_at_t = [y1 - y2 for y1, y2 in zip(self.full_model_at_t, self.first_base_model_at_t)]
   


    def update3(self, t, correction_window):
        """
        BAPC Step-3.
    
        Parameters:
        t (list): New time steps.
        correction_window (int): Number of previous time steps to consider for neighborhood.
        """
        
        # Reshape the observations for modification
        self.y_modified = np.reshape(self.y, (-1,1))
        
        # Determine the neighborhood based on correction_window parameter
        if correction_window == 0:
            self.neighborhood = []
        else:
            self.neighborhood = nbh.get_left_neighborhood_index(self.t, self.t[-1], correction_window)
    
        # Modify the observations in the neighborhood by subtracting the correction model predictions
        if len(self.neighborhood) > 0:
            correction_model_at_t = np.reshape(self.correction_model_at_t, (-1,1))
            self.y_modified[self.neighborhood] -= correction_model_at_t[self.neighborhood] 
    
        # Fit the second base model with the modified observations
        self.fit(
            model=self.second_base_model, 
            model_at_t=self.second_base_model_at_t, 
            tprev=self.t, 
            yprev=self.y_modified,
            predict_args=self.predict_args
        )
        
        # Append NaN values to the second_params_at_t dictionary for all but the last new time step
        for k in range(len(np.atleast_1d(t))-1):
            ut.append_dict(self.second_params_at_t, np.nan)
    
        # Append the current parameters of the second base model to the second_params_at_t dictionary
        ut.append_dict(self.second_params_at_t, self.second_base_model.params)
    
        # Update the surrogate model predictions
        self.surrogate_model_at_t = [y1 + (y1 - y2) for y1, y2 in zip(self.first_base_model_at_t, self.second_base_model_at_t)]
    
    def update(self, t=None, y=None, fit_residual=True, correction_window=0, fast=False, tol=0):
        """
        Update the models with new data and fit the residuals if required.
    
        Parameters:
        t (list, optional): New time steps. Defaults to None.
        y (list, optional): New observations. Defaults to None.
        fit_residual (bool, optional): Whether to fit the residuals. Defaults to True.
        correction_window (int, optional): Number of previous time steps to consider for neighborhood.
        fast (bool, optional): Whether to perform a fast update without fitting. Defaults to False.
        tol (float, optional): Tolerance for residual fitting. Defaults to 0.
        """
        
        # Ensure both t and y are provided or neither
        assert ((t is None) and (y is None)) or ((t is not None) and (y is not None)), 'Please provide both t and y or none of them'
    
        self.fit_residual = fit_residual
    
        if fast:
            # Perform a fast update without fitting
            self.update1_fast(t, y)
            self.update2_fast(fit_residual=fit_residual)
        else:
            # Perform a full update with fitting
            self.update1(t, y)
            self.update2(fit_residual=fit_residual, tol=tol)
    
        # Update the second base model and surrogate model
        self.update3(t, correction_window=correction_window) 
    
        # Store the current state for testing
        self.t_test = self.t.copy()
        self.first_base_model_at_t_test = self.first_base_model_at_t.copy()  
        self.correction_model_at_t_test = self.correction_model_at_t.copy()   
        self.full_model_at_t_test = self.full_model_at_t.copy()
        self.second_base_model_at_t_test  = self.second_base_model_at_t.copy()  
        self.surrogate_model_at_t_test = self.surrogate_model_at_t.copy()


    def predict(self, t):
        """
        Predict the different models output for the given time steps.
    
        Parameters:
        t (list or array): Time steps for which predictions are to be made.
        """
        
        # Reshape the input time steps for prediction
        t_test = np.atleast_2d(t).T
        
        # Predict using the first base model
        first_base_model_at_t_test = self.first_base_model.predict(t_test)
        
        # Predict using the correction model
        correction_model_at_t_test = self.correction_model.predict(t_test)
    
        if self.fit_residual:
            # Ensure the shapes of the first base model predictions and correction model predictions match
            sh1 = np.shape(first_base_model_at_t_test)
            sh2 = np.shape(self.correction_model.predict(t_test))
            assert sh1 == sh2, 'Implementation error: correction-model prediction and base-model prediction must have same shapes, {} not eq {}'.format(sh1, sh2)
            
            # Calculate the full model predictions by adding the correction model predictions to the first base model predictions
            full_model_at_t_test = first_base_model_at_t_test + self.correction_model.predict(t_test)
        else:
            # Use the correction model predictions as the full model predictions
            full_model_at_t_test = self.correction_model.predict(t_test)
    
        # Predict using the second base model
        second_base_model_at_t_test = self.second_base_model.predict(t_test)
        
        # Calculate the surrogate model predictions
        surrogate_model_at_t_test = 2 * first_base_model_at_t_test - second_base_model_at_t_test
    
        # Store the predictions for the given time steps
        self.t_test = list(t_test.flatten())
        self.first_base_model_at_t_test = list(first_base_model_at_t_test.flatten())
        self.correction_model_at_t_test = list(correction_model_at_t_test.flatten())
        self.full_model_at_t_test = list(full_model_at_t_test.flatten())
        self.second_base_model_at_t_test = list(second_base_model_at_t_test.flatten())
        self.surrogate_model_at_t_test = list(surrogate_model_at_t_test.flatten())


    def forecast(self, forecast_lenght=1):
        """
        Forecast the different models outputs for a given forecast length.
    
        Parameters:
        forecast_lenght (int, optional): Number of future time steps to forecast. Defaults to 1.
        """
        
        assert forecast_lenght >= 1, 'Assertion: forecast_lenght must be larger or equal than 1'
    
        # Forecast using the first base model and get the future time steps
        t_fore, first_base_model_at_t_fore = self.first_base_model.forecast(forecast_lenght, return_time_stamps=True)
        
        # Forecast using the correction model
        correction_model_at_t_fore = self.correction_model.forecast(forecast_lenght)
    
        if self.fit_residual:
            # Ensure the shapes of the first base model forecasts and correction model forecasts match
            sh1 = np.shape(first_base_model_at_t_fore)
            sh2 = np.shape(self.correction_model.forecast(forecast_lenght))
            if sh1 != sh2:
                warnings.warn('Implementation issue: correction-model prediction and base-model prediction should have same shapes, {} not eq {}'.format(sh1, sh2))
                # Calculate the full model forecasts by adding the correction model forecasts to the first base model forecasts
                full_model_at_t_fore = np.array([val1 + val2 for val1, val2 in zip(first_base_model_at_t_fore, self.correction_model.forecast(forecast_lenght))])
            else:
                full_model_at_t_fore = first_base_model_at_t_fore + self.correction_model.forecast(forecast_lenght)
        else:
            # Use the correction model forecasts as the full model forecasts
            full_model_at_t_fore = self.correction_model.forecast(forecast_lenght)
    
        # Forecast using the second base model
        second_base_model_at_t_fore = self.second_base_model.forecast(forecast_lenght)
        
        # Calculate the surrogate model forecasts
        surrogate_model_at_t_fore = 2 * first_base_model_at_t_fore - second_base_model_at_t_fore
    
        # Store the forecasts for the future time steps
        self.t_test = self.t + list(t_fore.flatten())
        self.first_base_model_at_t_test = self.first_base_model_at_t + list(first_base_model_at_t_fore.flatten())
        self.correction_model_at_t_test = self.correction_model_at_t + list(correction_model_at_t_fore.flatten())
        self.full_model_at_t_test = self.full_model_at_t + list(full_model_at_t_fore.flatten())
        self.second_base_model_at_t_test = self.second_base_model_at_t + list(second_base_model_at_t_fore.flatten())
        self.surrogate_model_at_t_test = self.surrogate_model_at_t + list(surrogate_model_at_t_fore.flatten())


    def explain(self, t=None, method='ig', **kwargs):
        """
        Explain the model predictions using the specified method.
    
        Parameters:
        t (list or array, optional): Time steps for which explanations are to be made. Defaults to None.
        method (str, optional): Explanation method to use ('ig' for Integrated Gradients, 'lime' for LIME). Defaults to 'ig'.
        **kwargs: Additional arguments for the explanation method.
    
        Returns:
        Explanation results based on the specified method.
        """
        
        if method == 'ig':
            # Get the components of the model for Integrated Gradients explanation
            components = self.get_components()
            return pitsa.xai.importance.explain_ig(components, t, **kwargs)
        elif method == 'lime':
            if self.correction_model.fit_flag:
                xcoefs = []
                predict_fn = lambda x: np.array(self.correction_model.predict_from_array(np.transpose(x)).flatten())
                for k in np.atleast_1d(t):
                    ar_order = self.correction_model.ar_order
                    sample = np.reshape(self.y[k-ar_order:k], (-1,1))
                    xcoef = pitsa.xai.importance.explain_lime(sample, predict_fn, **kwargs)
                    xcoefs.append(xcoef)
                xcoefs = np.concatenate(xcoefs, axis=1).T
            else:
                print('Warning: correction model is not fitted, returning zeros')
                xcoefs = np.zeros((len(np.atleast_1d(t)), self.correction_model.ar_order))
            return xcoefs
        else:
            raise ValueError('Unknown method: {}'.format(method))



    def get_params_as_df(self, keys=None, t=None, transform=None, decimals=None, long=False):
        """
        Get the model parameters as a DataFrame.
    
        Parameters:
        keys (list, optional): Specific keys to include in the DataFrame. Defaults to None.
        t (list, optional): Time steps to include in the DataFrame. Defaults to None.
        transform (callable, optional): Transformation function to apply to the parameters. Defaults to None.
        decimals (int, optional): Number of decimal places to round the parameters. Defaults to None.
        long (bool, optional): Whether to return the DataFrame in long format. Defaults to False.
    
        Returns:
        DataFrame: Model parameters as a DataFrame.
        """
        components = self.get_components()
        return ut.get_params_as_df(components, keys=keys, t=t, transform=transform, decimals=decimals, long=long)
    
    def get_components(self):
        """
        Get the components of the model.
    
        Returns:
        dict: Dictionary of model components.
        """
        # Collect components that are arrays, lists, or dictionaries
        d = {key: self.__dict__[key] for key in self.__dict__
             if isinstance(self.__dict__[key], np.ndarray) or 
             isinstance(self.__dict__[key], list) or 
             isinstance(self.__dict__[key], dict)}
        
        # Add the first base model's name and initial conditions (ics) if available
        if hasattr(self.first_base_model, 'ics'):
            ics = self.first_base_model.ics
        else:
            ics = None
            
        d['first_base_model'] = {"name": self.first_base_model.name, "ics": ics}
    
        return d
    
    def describe_components(self):
        """
        Print the shapes of the model components.
        """
        d = self.get_components()
        for key in d:
            print(key, np.shape(d[key]))
    
    
    def plot(self, 
             ax=None, 
             **kwargs):
        """
        Plot the BAPC.
    
        Parameters:
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto. Defaults to None.
        **kwargs: Additional arguments for the plotting function.
    
        Returns:
        matplotlib.axes.Axes: Axes object with the plot.
        """
        
        # Get the components of the model
        components = self.get_components()
    
        # Plot the components using the specified plotting function
        ax = pitsa.graphics.pltbapc.plot(components, 
                                         ax=ax,
                                         **kwargs)
        
        return ax
        
    
