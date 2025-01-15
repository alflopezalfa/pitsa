import pandas as pd


def append_dict(d, other):
    """
    Append values from another dictionary or a single value to the lists in the given dictionary.

    Parameters:
    d (dict): The dictionary to which values will be appended.
    other (dict or any): The dictionary with values to append or a single value to append to all keys.
    """
    for key in d:
        if isinstance(other, dict):
            # Append the value from the other dictionary to the corresponding list in d
            d[key].append(other[key])
        else:
            # Append the single value to all lists in d
            d[key].append(other)




def get_params_as_df(components, keys=None, t=None, transform=None, decimals=None, long=False):
    """
    Get the model parameters as a DataFrame.

    Parameters:
    components (dict): Dictionary of model components.
    keys (list, optional): Specific keys to include in the DataFrame. Defaults to None.
    t (list, optional): Time steps to include in the DataFrame. Defaults to None.
    transform (callable, optional): Transformation function to apply to the parameters. Defaults to None.
    decimals (int, optional): Number of decimal places to round the parameters. Defaults to None.
    long (bool, optional): Whether to return the DataFrame in long format. Defaults to False.

    Returns:
    DataFrame: Model parameters as a DataFrame.
    """
    
    if keys is None:
        keys = list(components['first_params_at_t'].keys())

    # Extract the first and second parameters at time t
    first_params_at_t = {key: components['first_params_at_t'][key] for key in keys}
    second_params_at_t = {key: components['second_params_at_t'][key] for key in keys}

    # Create DataFrames from the parameters
    df1 = pd.DataFrame.from_dict(first_params_at_t) 
    df2 = pd.DataFrame.from_dict(second_params_at_t) 
    df3 = df1 - df2

    # Rename columns to indicate before, after, and delta (difference)
    df1.rename(columns={key: '{}_before'.format(key) for key in first_params_at_t}, inplace=True)
    df2.rename(columns={key: '{}_after'.format(key) for key in second_params_at_t}, inplace=True)
    df3.rename(columns={key: '{}_delta'.format(key) for key in first_params_at_t}, inplace=True)

    # Concatenate the DataFrames
    df = pd.concat([df1, df2, df3], axis=1)
    df.index = components['t']

    # Apply transformation if provided
    if transform is not None:
        for key_new in transform.keys():
            key = transform[key_new][0]
            fun = transform[key_new][1]
            df['{}_before'.format(key_new)] = df.apply(lambda x: fun(x['{}_before'.format(key)]), axis=1)
            df['{}_after'.format(key_new)] = df.apply(lambda x: fun(x['{}_after'.format(key)]), axis=1)
            df['{}_delta'.format(key_new)] = df['{}_before'.format(key_new)] - df['{}_after'.format(key_new)]
            keys.append(key_new)

    nobs = df.index[-1] 

    if t is None:
        t = nobs

    # Filter the DataFrame to include only the specified time steps
    df = df.loc[t:nobs]

    # Round the DataFrame to the specified number of decimal places
    if decimals is not None:
        df = df.round(decimals)
        for key in keys:
            df['{}_delta'.format(key)] = df['{}_before'.format(key)] - df['{}_after'.format(key)] 

    df.index.name = 'time'

    # Convert the DataFrame to long format if specified
    if long:
        df.reset_index(inplace=True)
        df = pd.wide_to_long(df, i='time', j='step', stubnames=keys, sep='_', suffix=r'\w+')
        df.sort_index(level=0, sort_remaining=False, inplace=True)

    return df


