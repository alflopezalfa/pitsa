import numpy as np
import pandas as pd
from scipy.special import binom
from mpmath import mp, binomial, power
import pitsa
import sys
import warnings
import tsmule



def explain_ig(components, t=None, **kwargs):
    """
    Explain the model predictions using Integrated Gradients (IG).

    Parameters:
    components (dict): Dictionary of model components.
    t (list or array, optional): Time steps for which explanations are to be made. Defaults to None.
    **kwargs: Additional arguments for the Integrated Gradients explanation method.

    Returns:
    tuple: Explanation and importance DataFrames.
    """
    
    # Extract the last parameters before and after the update
    theta_bf = {key: val[-1] for key, val in components["first_params_at_t"].items()}
    theta_af = {key: val[-1] for key, val in components["second_params_at_t"].items()}

    if t is None:
        t = components['t_test']

    # Get initial conditions (ics) and model name
    model = components["first_base_model"].get('name', None)
    ics = components["first_base_model"].get('ics', None)

    # Compute the explanation and importance using Integrated Gradients
    explanation, importance = pitsa.xai.importance.integrated_gradient(t, 
                                                                        theta_bf,
                                                                        theta_af, 
                                                                        model,
                                                                        ics=ics,
                                                                        **kwargs)
    # Set the index of the explanation to the last time step
    explanation.index = components["t"][-1:]

    return explanation, importance


def integrated_gradient(t, theta_bf, theta_af={}, model=None, **kwargs):
    """
    Compute the Integrated Gradients (IG) for the given model and parameters.

    Parameters:
    t (array): Time steps for which explanations are to be made.
    theta_bf (dict): Parameters before the update.
    theta_af (dict, optional): Parameters after the update. Defaults to an empty dictionary.
    model (str, optional): Model name. Defaults to None.
    **kwargs: Additional arguments for specific models.

    Returns:
    tuple: Explanation and importance DataFrames.
    """
    
    if model == 'Sklearn-Model':
        t = np.atleast_1d(t)
        explanation, importance = ig_poly(t, theta_bf, theta_af)
    elif model == 'KernelBasisFunction':
        t = np.atleast_1d(t)
        explanation, importance = ig_poly_sin(t, theta_bf, theta_af)
    elif model == 'Cosine':
        t = np.atleast_1d(t)
        explanation, importance = ig_sinusoidal(t, theta_bf, theta_af)
    elif model == 'Sinusoidal':
        t = np.atleast_1d(t)
        explanation, importance = ig_poly_sin(t, theta_bf, theta_af)
    elif model == 'RobustAutoRegressiveRegression':
        ics = kwargs.get('ics', None)
        hp = kwargs.get('hp', False)
        time_start = kwargs.get('time_start', 0)
        precision = kwargs.get('precision', 50)
        assert ics is not None, 'Initial conditions are required for AR(2) model'
        y1 = ics[0]
        y2 = ics[1]
        explanation, importance = ig_ar(t, theta_bf, theta_af, y1, y2, hp, time_start, precision)
    else:
        raise NotImplementedError(f'Model {model} not implemented')
    
    return explanation, importance


def ig_poly(t, theta_bf, theta_af):
    """
    Compute the Integrated Gradients (IG) for a polynomial model.

    Parameters:
    t (array): Time steps for which explanations are to be made.
    theta_bf (dict): Parameters before the update.
    theta_af (dict): Parameters after the update.

    Returns:
    tuple: Explanation and importance DataFrames.
    """
    
    # Extract parameters before and after the update
    intercept_bf = theta_bf.get('intercept', np.nan)
    slope_bf = theta_bf.get('slope', np.nan)
    quadratic_bf = theta_bf.get('quadratic', np.nan)

    if theta_af == {}:
        intercept_af = 0
        slope_af = 0
        quadratic_af = 0
    else:
        intercept_af = theta_af.get('intercept', np.nan)
        slope_af = theta_af.get('slope', np.nan)
        quadratic_af = theta_af.get('quadratic', np.nan)

    # Calculate the differences (deltas) between the parameters
    delta_intercept = intercept_bf - intercept_af
    delta_slope = slope_bf - slope_af
    delta_quadratic = quadratic_bf - quadratic_af

    # Create a delta array and feature matrix
    delta = np.array([[delta_intercept, delta_slope, delta_quadratic]]).T
    features = np.vstack((np.ones(len(t)), t, t**2))
    
    # Compute the output by multiplying deltas with features
    out = delta * features

    # Create DataFrames for the explanation and importance
    out1 = pd.DataFrame(delta.T, columns=['intercept', 'slope', 'quadratic'])
    out2 = pd.DataFrame(out.T, index=t, columns=['intercept', 'slope', 'quadratic'])

    # drop nan columns
    out1 = out1.dropna(axis=1)
    out2 = out2.dropna(axis=1)

    return out1, out2


def ig_sinusoidal(t, theta_bf, theta_af):
    """
    Compute the Integrated Gradients (IG) for a sinusoidal model.

    Parameters:
    t (array): Time steps for which explanations are to be made.
    theta_bf (dict): Parameters before the update.
    theta_af (dict): Parameters after the update.

    Returns:
    tuple: Explanation and importance DataFrames.
    """
    
    # Extract parameters before and after the update
    amplitude_bf = theta_bf.get('amplitude', 0)
    decay_bf = theta_bf.get('decay', 0)
    frequency_bf = theta_bf.get('frequency', 0)
    phase_bf = theta_bf.get('phase', 0)
   
    amplitude_af = theta_af.get('amplitude', 0)
    decay_af = theta_af.get('decay', 0)
    frequency_af = theta_af.get('frequency', 0)
    phase_af = theta_af.get('phase', 0)

    # Calculate the differences (deltas) between the parameters
    delta_amplitude = amplitude_bf - amplitude_af
    delta_decay = decay_bf - decay_af
    delta_frequency = frequency_bf - frequency_af
    delta_phase = phase_bf - phase_af

    # Create a delta array
    delta = np.array([[delta_amplitude, delta_decay, delta_frequency, delta_phase]])

    if delta_decay == 0 and delta_frequency == 0 and delta_phase == 0:
        # If there are no changes in decay, frequency, and phase, compute IG directly
        ig = np.vstack((delta_amplitude * np.ones(len(t)), np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))))
    else:
        # Compute the sinusoidal functions before and after the update
        g_bf = np.exp(-decay_bf * t + 1j * frequency_bf * t + 1j * phase_bf)
        g_af = np.exp(-decay_af * t + 1j * frequency_af * t + 1j * phase_af)
        f_bf = amplitude_bf * g_bf
        f_af = amplitude_af * g_af
        delta_f = f_bf - f_af

        # Compute the denominators and weights for IG
        den = (-delta_decay * t + 1j * delta_frequency * t + 1j * delta_phase)
        delta_g = (g_bf - g_af) / den
        w_decay = -delta_decay * t / den
        w_frequency = 1j * delta_frequency * t / den
        w_phase = 1j * delta_phase / den

        # Compute the initial IG and final IG
        ig0 = delta_amplitude * delta_g
        ig = np.vstack((ig0, w_decay * (delta_f - ig0), w_frequency * (delta_f - ig0), w_phase * (delta_f - ig0)))
        ig = np.real(ig)

    # Create DataFrames for the explanation and importance
    out1 = pd.DataFrame(delta, columns=['amplitude', 'decay', 'frequency', 'phase'])
    out2 = pd.DataFrame(ig.T, index=t, columns=['amplitude', 'decay', 'frequency', 'phase'])

    return out1, out2


def ig_poly_sin(t, theta_bf, theta_af):
    """
    Compute the Integrated Gradients (IG) for a model combining polynomial and sinusoidal components.

    Parameters:
    t (array): Time steps for which explanations are to be made.
    theta_bf (dict): Parameters before the update.
    theta_af (dict): Parameters after the update.

    Returns:
    tuple: Explanation and importance DataFrames.
    """
    
    # Compute IG for the polynomial component
    explanation_poly, importance_poly = ig_poly(t, theta_bf, theta_af)
    
    # Compute IG for the sinusoidal component
    explanation_sinusoidal, importance_sinusoidal = ig_sinusoidal(t, theta_bf, theta_af)
    
    # Concatenate the explanations and importances
    explanation = pd.concat([explanation_poly, explanation_sinusoidal], axis=1)
    importance = pd.concat([importance_poly, importance_sinusoidal], axis=1)

    return explanation, importance


####### AR(2) model #############

def _gamma(a, da, h):
    """
    Compute the segment starting at a with direction da, evaluated at h.

    Parameters:
    a (float): Initial value.
    da (float): Change in value.
    h (float): Step size.

    Returns:
    float: Intermediate gamma value.
    """
    return a + da*h

def gamma(m, a, da, h):
    """
    Compute the gamma value raised to the power of m.

    Parameters:
    m (int): Power to which the gamma value is raised.
    a (float): Initial value.
    da (float): Change in value.
    h (float): Step size.

    Returns:
    float: Gamma value raised to the power of m.
    """
    return np.power(_gamma(a, da, h), m)

def gamma_der(m, a, da, h):
    """
    Compute the derivative of gamma raised to the power of m, respect to h.

    Parameters:
    m (int): Power to which the gamma value is raised.
    a (float): Initial value.
    da (float): Change in value.
    h (float): Step size.

    Returns:
    float: Derivative of the gamma value raised to the power of m.
    """
    return m*np.power(_gamma(a, da, h), m-1)*da

def _Phi(t, phi):
    """
    Compute the Phi function for the given time step and parameters.

    Parameters:
    t (int): Time step.
    phi (array): Array of parameters.

    Returns:
    float: Computed Phi value.
    """
    m = np.floor(t/2)
    k = np.arange(0, m+1)
    s = binom(t-k, k)*np.power(phi[0], t-2*k)*np.power(phi[1], k)
    return np.sum(s)

def Phi1(t, phi):
    """
    Compute the Phi1 function for the given time step and parameters.

    Parameters:
    t (int): Time step.
    phi (array): Array of parameters.

    Returns:
    float: Computed Phi1 value.
    """
    t = t + 1 # like in the paper, time-series index starts from 1
    if t == 1:
        return 0
    elif t == 2:
        return 1
    else: 
        return _Phi(t-2, phi)
    
def Phi2(t, phi):
    """
    Compute the Phi2 function for the given time step and parameters.

    Parameters:
    t (int): Time step.
    phi (array): Array of parameters.

    Returns:
    float: Computed Phi2 value.
    """
    t = t + 1 # like in the paper, time-series index starts from 1
    if t == 1:
        return 1
    elif t == 2:
        return 0
    else: 
        return phi[1]*_Phi(t-3, phi)
    
def ar(t, phi, ics):
    """
    Compute the autoregressive (AR) model output for the given time step and parameters.
    Based on Proposition 3 in the paper.

    Parameters:
    t (int): Time step.
    phi (array): Array of AR parameters.
    ics (array): Initial conditions [y1, y2].

    Returns:
    float: AR model output.
    """
    y1 = ics[0]
    y2 = ics[1]
    return Phi1(t, phi)*y2 + Phi2(t, phi)*y1

    
def _bar_Phi1(t, j, phi, dphi):
    """
    Auxiliar for computing the \bar{Phi}_1 in Proposition 5-part 3.

    Parameters:
    t (int): Time step.
    j (int): Index associated to autoregressive lag (j=0,1).
    phi (array): Array of parameters.
    dphi (array): Array of changes in parameters.

    Returns:
    float: Computed \bar{Phi}_1 value.
    """
    m = np.floor(t/2) 
    k = np.arange(0, m+1)
    b = binom(t-k, k)*(t-2*k)
    h = np.linspace(0, 1, 101)
    h, k = np.meshgrid(h, k)
    gh = gamma(t-1-2*k, phi[0], dphi[0], h)*gamma(k+j, phi[1], dphi[1], h)
    s = b*np.mean(gh, axis=1)
    return dphi[0]*np.sum(s)

def _bar_Phi2(t, j, phi, dphi):
    """
    Auxiliar for computing the \bar{Phi}_2 in Proposition 5-part 3.

    Parameters:
    t (int): Time step.
    j (int): Index associated to autoregressive lag (j=0,1).
    phi (array): Array of parameters.
    dphi (array): Array of changes in parameters.

    Returns:
    float: Computed bar_Phi2 value.
    """
    m = np.floor(t/2) 
    k = np.arange(0, m+1)
    b = binom(t-k, k)*(k+j)
    h = np.linspace(0, 1, 101)
    h, k = np.meshgrid(h, k)
    gh = gamma(t-2*k, phi[0], dphi[0], h)*gamma(k-1+j, phi[1], dphi[1], h)
    s = b*np.mean(gh, axis=1)
    return dphi[1]*np.sum(s)

def bar_Phi1(t, phi, dphi, y):
    """
    Compute \bar{Phi}_1 in Proposition 5-part 3.

    Parameters:
    t (int): Time step.
    phi (array): Array of parameters.
    dphi (array): Array of changes in parameters.
    y (array): Initial conditions [y1, y2].

    Returns:
    float: Computed \bar{Phi}_1 value.
    """
    t = t + 1 # like in the paper, time-series index starts from 1
    if t == 1:
        return 0
    elif t == 2:
        return 0
    else: 
        return _bar_Phi1(t-2, 0, phi, dphi)*y[1] + _bar_Phi1(t-3, 1, phi, dphi)*y[0]
    
def bar_Phi2(t, phi, dphi, y):
    """
    Compute \bar{Phi}_2 in Proposition 5-part 3.

    Parameters:
    t (int): Time step.
    phi (array): Array of parameters.
    dphi (array): Array of changes in parameters.
    y (array): Initial conditions [y1, y2].

    Returns:
    float: Computed \bar{Phi}_2 value.
    """
    t = t + 1 # like in the paper, time-series index starts from 1
    if t == 1:
        return 0
    elif t == 2:
        return 0
    else: 
        return _bar_Phi2(t-2, 0, phi, dphi)*y[1] + _bar_Phi2(t-3, 1, phi, dphi)*y[0]
    
def ig_ar(t, theta_bf, theta_af, y1, y2, hp=False, time_start=0, precision=50):
    """
    Compute the Integrated Gradients (IG) for an autoregressive (AR) model.
    Based on Proposition 5 - part 3, in the paper.

    Parameters:
    t (int): Time step for which explanations are to be made.
    theta_bf (dict): Parameters before the update.
    theta_af (dict): Parameters after the update.
    y1 (float): Initial condition y1.
    y2 (float): Initial condition y2.
    hp (bool, optional): Whether to use high precision computation. Defaults to False.
    time_start (int, optional): Starting time step. Defaults to 0.
    precision (int, optional): Precision for high precision computation. Defaults to 50.

    Returns:
    tuple: Explanation and importance DataFrames.
    """
    phi1_bf = theta_bf['ar.L1']
    phi2_bf = theta_bf['ar.L2']
    phi1_af = theta_af['ar.L1']
    phi2_af = theta_af['ar.L2']
    delta = np.array([[phi1_bf - phi1_af, phi2_bf - phi2_af]])

    phi = [phi1_af, phi2_af]
    dphi = [phi1_bf - phi1_af, phi2_bf - phi2_af]

    if hp:
        # Compute IG using high precision method
        ig = _ig_ar_hp(t-time_start, phi, dphi, [y1, y2], precision)
    else:
        # Compute IG using standard method
        ig1 = bar_Phi1(t, phi, dphi, [y1, y2])
        ig2 = bar_Phi2(t, phi, dphi, [y1, y2])
        ig = np.array([[ig1, ig2]])
  
    # Create DataFrames for the explanation and importance
    out1 = pd.DataFrame(delta, columns=['ar.L1', 'ar.L2'])
    out2 = pd.DataFrame(ig, index=[t], columns=['ar.L1', 'ar.L2'])

    return out1, out2


def _ig_ar_hp(t, phi, dphi, ics, precision=50):
    """
    Compute the integrated gradients for the AR model with high precision using mpmath.

    Parameters:
    - t (int): Time index.
    - phi (array-like): Model parameters.
    - dphi (array-like): Derivatives of model parameters.
    - ics (array-like): Initial conditions.
    - precision (int): Decimal places for precision.

    Returns:
    - np.ndarray: Array containing ig1 and ig2 values.
    """
    print('Computing IG for AR model with high precision')

    # Set the precision for mpmath
    print(f'Precision: {precision}')
    mp.dps = precision  # Decimal places for precision

    # Initial conditions
    y1 = mp.mpf(ics[0])
    y2 = mp.mpf(ics[1])
    phi1 = mp.mpf(phi[0])
    phi2 = mp.mpf(phi[1])
    dphi1 = mp.mpf(dphi[0])
    dphi2 = mp.mpf(dphi[1])

    # Compute gamma and gamma_der functions inline
    def gamma(m, a, da, h):
        return power(a + da * h, m)
    
    def gamma_der(m, a, da, h):
        return m * power(a + da * h, m - 1) * da
    
    # Compute the binomial coefficients
    m = mp.ceil(t / 2) - 1
    k = np.arange(0, int(m) + 1)
    t = t - 1
    gh11 = [binomial(t - kk, kk) for kk in k]
    gh21 = [binomial(t - kk, kk) for kk in k]
    gh12 = [binomial(t - 1 - kk, kk) for kk in k]
    gh22 = [binomial(t - 1 - kk, kk) for kk in k]

    # Compute the integrands

    gh11 = [gh11[i] * mp.quad(lambda h: gamma_der(t - 2 * k[i], phi1, dphi1, h) * gamma(k[i], phi2, dphi2, h), [0, 1]) for i in range(len(k))]
    ig11 = mp.fsum(gh11)

    gh21 = [gh21[i] * mp.quad(lambda h: gamma(t - 2 * k[i], phi1, dphi1, h) * gamma_der(k[i], phi2, dphi2, h), [0, 1]) for i in range(len(k))]
    ig21 = mp.fsum(gh21)

    gh12 = [gh12[i] * mp.quad(lambda h: gamma_der(t - 1 - 2 * k[i], phi1, dphi1, h) * gamma(k[i] + 1, phi2, dphi2, h), [0, 1]) for i in range(len(k))]
    ig12 = mp.fsum(gh12)

    gh22 = [gh22[i] * mp.quad(lambda h: gamma(t - 1 - 2 * k[i], phi1, dphi1, h) * gamma_der(k[i] + 1, phi2, dphi2, h), [0, 1]) for i in range(len(k))]
    ig22 = mp.fsum(gh22)

    ig1 = ig11 * y2 + ig12 * y1
    ig2 = ig21 * y2 + ig22 * y1
    ig = np.array([[float(ig1), float(ig2)]])
    
    return ig


def explain_lime(sample, predict_fn, warning=False, seed=42, **kwargs):
    """
    Explain the model predictions using LIME (Local Interpretable Model-agnostic Explanations).

    Parameters:
    sample (array): Input sample for which explanations are to be made.
    predict_fn (callable): Prediction function of the model.
    warning (bool, optional): Whether to show warnings in case of errors. Defaults to False.
    **kwargs: Additional arguments for the LIME explanation method.

    Returns:
    array: LIME coefficients.
    """
    
    n_samples = kwargs.get('n_samples', 20)
    partitions = kwargs.get('partitions', 10)
    win_length = kwargs.get('win_length', 3)

    # Initialize the segmenter and explainer for LIME
    segmenter = tsmule.sampling.segment.WindowSegmentation(partitions=partitions, win_length=win_length)
    explainer = tsmule.xai.lime.LimeTS(n_samples=n_samples, replace_method='global_mean', segmenter=segmenter)

    try:
        # Compute the LIME explanation
        np.random.seed(seed)
        xcoef = explainer.explain(sample, predict_fn, segmentation_method='uniform')
    except:
        if warning:
            warnings.warn('Error in LIME explanation')
        # Return NaN values in case of errors
        xcoef = np.full(sample.shape, np.nan)

    return xcoef







