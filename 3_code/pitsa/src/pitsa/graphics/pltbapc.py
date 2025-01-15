import matplotlib.pyplot as plt
import seaborn as sns


# From metropolis.mplstyle
colors = ['#4C72B0', '#EB811B', '#14B03D', '#604C38']
palette = sns.color_palette(sns.set_palette(sns.color_palette(colors)))
c1 = palette[0]
c11 = 'lightskyblue'
c2 = palette[2]
c22 = 'lightgreen'
c3 = palette[1]
c4 = palette[3]


def _plot(x, 
          y,
          second_base_model_at_x, 
          surrogate_model_at_x, 
          neighborhood,  
          ax=None, 
          **kwargs):
    """
    Internal function to plot the model components.

    Parameters:
    x (array): Time steps.
    y (array): Observations.
    second_base_model_at_x (array): Predictions of the second base model.
    surrogate_model_at_x (array): Predictions of the surrogate model.
    neighborhood (list): Indices of the neighborhood.
    ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto. Defaults to None.
    **kwargs: Additional arguments for the plotting function.

    Returns:
    matplotlib.axes.Axes: Axes object with the plot.
    """
    
    # Calculate the forecast horizon
    fh = len(x) - len(y)
    
    # Create a new figure and axes if not provided
    if ax is None:
        plt.figure(**kwargs)
        ax = plt.gca()

    # Plot the observations
    ax.plot(x[:len(y)], 
            y, 
            color='tab:blue', 
            zorder=0, 
            label=r'$y$')
    
    # Plot the second base model predictions
    ax.plot(
        x[neighborhood[0]:neighborhood[-1]+1+fh], 
        second_base_model_at_x[neighborhood[0]:neighborhood[-1]+1+fh], 
        color=c4, 
        zorder=1, 
        label=r'$f_{\theta_r}$'
    )
    
    # Plot the surrogate model predictions
    ax.plot(
        x[neighborhood[0]:neighborhood[-1]+1+fh], 
        surrogate_model_at_x[neighborhood[0]:neighborhood[-1]+1+fh], 
        color=c3, 
        zorder=2, 
        label=r'$f_0 + \Delta f_r$'
    )

    return ax

def plot(components, 
         ax=None, 
         **kwargs):
    """
    Plot the model components.

    Parameters:
    components (dict): Dictionary of model components.
    ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto. Defaults to None.
    **kwargs: Additional arguments for the plotting function.

    Returns:
    matplotlib.axes.Axes: Axes object with the plot.
    """
    
    ax = _plot(x=components['t_test'],  
               y=components['y'],
               second_base_model_at_x=components['second_base_model_at_t_test'], 
               surrogate_model_at_x=components['surrogate_model_at_t_test'], 
               neighborhood=components['neighborhood'],
               ax=ax, 
               **kwargs)

    return ax
