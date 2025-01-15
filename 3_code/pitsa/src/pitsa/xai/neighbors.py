
def get_left_neighborhood_index(x, x_instance, correction_window):
    """
    Get the indices of the left neighborhood of a given instance.

    Parameters:
    x (array-like): Array of time steps or values.
    x_instance (scalar): The instance for which the neighborhood is to be found.
    correction_window (int): Number of previous time steps to consider for the neighborhood.

    Returns:
    list: Indices of the left neighborhood.
    """

    # Find indices where the difference between x_instance and elements in x is non-negative
    N = [i for i, xi in enumerate(x) if 0.0 <= x_instance - xi]
    
    # Return the last 'look_back' indices
    N = N[-correction_window:]
    return N