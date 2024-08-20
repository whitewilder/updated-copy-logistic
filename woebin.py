# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import re
import warnings
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import os
import platform
from .condition_fun import *
from .info_value import *

# Convert vector to DataFrame, splitting rows with '%,%' and replacing 'missing' with np.nan
def vector_to_dataframe(vector):
    '''
    Create a DataFrame based on the provided vector. 
    Split rows containing '%,%' into multiple rows. 
    Replace 'missing' with np.nan.
    
    Parameters
    ----------
    vector : list
        List of values to be converted.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with three columns: {'original_value': original vector, 'index': index of vector, 'split_value': split vector}
    '''
    if vector is not None:
        vector = [str(i) for i in vector]
        df_original = pd.DataFrame({'original_value': vector}).assign(index=lambda x: x.index)
        df_split = pd.DataFrame([i.split('%,%') for i in vector], index=vector) \
            .stack().replace('missing', np.nan) \
            .reset_index(name='split_value') \
            .rename(columns={'level_0': 'original_value'})[['original_value', 'split_value']]
        
        return pd.merge(df_original, df_split, on='original_value')


def handle_missing_special_values(df_melted, breakpoints, special_values):
    '''
    Add 'missing' to special_values if there is np.nan in df_melted['split_value']
    and 'missing' is not specified in breakpoints or special_values.
    
    Parameters
    ----------
    df_melted : pandas.DataFrame
        Melted DataFrame.
    breakpoints : list
        List of breakpoints.
    special_values : list
        List of special values.
    
    Returns
    -------
    list
        Updated list of special values.
    '''
    if df_melted['split_value'].isnull().any():
        if breakpoints is None or not any('missing' in str(i) for i in breakpoints):
            if special_values is None:
                special_values = ['missing']
            elif not any('missing' in str(i) for i in special_values):
                special_values = ['missing'] + special_values
                
    return special_values


# Count the number of occurrences of 0 and 1 in a series
def count_zeros(series): return sum(series == 0)
def count_ones(series): return sum(series == 1)

def dtm_binning_with_special_values(melted_df, breaks, special_values):
    '''
    Split the original melted DataFrame into:
    - binning_sv: Binning of special values
    - melted_df: DataFrame without special values
    
    Parameters
    ----------
    melted_df : pandas.DataFrame
        Melted DataFrame with at least 'value' column.
    breaks : list
        List of breakpoints for binning.
    special_values : list
        List of special values to be separated.
    
    Returns
    -------
    dict
        A dictionary with two keys:
        - 'binning_sv': DataFrame with binned special values.
        - 'melted_df': DataFrame without special values.
    '''
    # Ensure missing values are handled correctly in special values
    special_values = handle_missing_special_values(melted_df, breaks, special_values)
    
    if special_values is not None:
        # Convert special values to DataFrame
        sv_df = vector_to_dataframe(special_values)
        
        # Handle numeric dtype for value column
        if is_numeric_dtype(melted_df['value']):
            sv_df['value'] = sv_df['value'].astype(melted_df['value'].dtype)
            sv_df['bin_chr'] = np.where(
                sv_df['value'].isna(), sv_df['bin_chr'],
                sv_df['value'].astype(melted_df['value'].dtype).astype(str)
            )
        
        # Determine the index name
        if None in melted_df.index.names:
            if len(melted_df.index.names) == 1:
                melted_df_index = 'index'
            elif len(melted_df.index.names) > 1:
                raise ValueError("MultiIndex contains NoneType")
        else:
            melted_df_index = melted_df.index.names
        
        # Split melted_df into special value and remaining
        special_values_df = pd.merge(
            melted_df.reset_index().fillna("missing"),
            sv_df[['value']].fillna("missing"),
            how='inner', on='value'
        ).set_index(melted_df_index)
        
        remaining_df = melted_df[~melted_df.index.isin(special_values_df.index)].reset_index() if len(special_values_df.index) < len(melted_df.index) else None
        
        if special_values_df.shape[0] == 0:
            return {'binning_sv': None, 'melted_df': remaining_df}
        
        # Create binning of special values
        binning_sv = pd.merge(
            special_values_df.fillna('missing')
            .groupby(['variable', 'value'], group_keys=False)['y']
            .agg([count_zeros, count_ones])
            .reset_index()
            .rename(columns={'count_zeros': 'good', 'count_ones': 'bad'}),
            sv_df.fillna('missing'),
            on='value'
        ).groupby(['variable', 'rowid', 'bin_chr'], group_keys=False)
         .agg({'bad': sum, 'good': sum})
         .reset_index()
         .rename(columns={'bin_chr': 'bin'})
         .drop('rowid', axis=1)
        
    else:
        binning_sv = None
    
    return {'binning_sv': binning_sv, 'melted_df': remaining_df}

def check_and_correct_empty_bins(melted_df, binning):
    '''
    Check and correct for empty bins in the DataFrame for numeric variables.
    
    Parameters
    ----------
    melted_df : pandas.DataFrame
        DataFrame containing bins and values.
    binning : pandas.DataFrame
        DataFrame with binning information.
    
    Returns
    -------
    pandas.DataFrame
        Updated binning DataFrame.
    '''
    bin_list = np.unique(melted_df['bin'].astype(str)).tolist()
    if 'nan' in bin_list:
        bin_list.remove('nan')
    
    # Extract bin boundaries
    bin_left = set(re.match(r'\[(.+),(.+)\)', i).group(1) for i in bin_list if re.match(r'\[(.+),(.+)\)', i))
    bin_right = set(re.match(r'\[(.+),(.+)\)', i).group(2) for i in bin_list if re.match(r'\[(.+),(.+)\)', i))
    
    if bin_left != bin_right:
        # Create new breakpoints and labels
        breakpoints = sorted(map(float, ['-inf'] + list(bin_right) + ['inf']))
        labels = ['[{},{})'.format(breakpoints[i], breakpoints[i+1]) for i in range(len(breakpoints) - 1)]
        
        # Update bins in melted_df DataFrame
        melted_df['bin'] = pd.cut(melted_df['value'], breakpoints, right=False, labels=labels)
        binning = melted_df.groupby(['variable', 'bin'], group_keys=False)['y'] \
            .agg([count_zeros, count_ones]) \
            .reset_index() \
            .rename(columns={'count_zeros': 'good', 'count_ones': 'bad'})
        
        # Warning message about updated breakpoints
        warnings.warn(
            "Break points have been modified to '[{}]'. There were empty bins based on the provided break points.".format(
                ','.join(bin_right))
        )
    
    return binning

def woebin2_breaks(melted_df, breaks, special_values):
    '''
    Get binning if breaks are provided.
    
    Parameters
    ----------
    melted_df : pandas.DataFrame
        Melted DataFrame containing the values to be binned.
    breaks : list
        List of breakpoints for binning.
    special_values : list
        List of special values to be considered separately.
    
    Returns
    -------
    dict
        A dictionary with two keys:
        - 'binning_sv': DataFrame with special values binning.
        - 'binning': DataFrame with binning based on provided breaks.
    '''
    
    # Convert breaks from vector to DataFrame
    breakpoints_df = split_vec_todf(breaks)
    
    # Split melted_df into binning of special values and the rest
    binning_result = dtm_binning_with_special_values(melted_df, breaks, special_values)
    remaining_df = binning_result['melted_df']
    special_values_binning = binning_result['binning_sv']
    
    if remaining_df is None:
        return {'binning_sv': special_values_binning, 'binning': None}
    
    # Binning based on numeric values
    if is_numeric_dtype(remaining_df['value']):
        # Determine the best breakpoints
        best_breakpoints = ['-inf'] + list(set(breakpoints_df['value'].tolist()).difference(set([np.nan, '-inf', 'inf', 'Inf', '-Inf']))) + ['inf']
        best_breakpoints = sorted(map(float, best_breakpoints))
        
        # Create labels for bins
        labels = ['[{},{})'.format(best_breakpoints[i], best_breakpoints[i+1]) for i in range(len(best_breakpoints) - 1)]
        remaining_df['bin'] = pd.cut(remaining_df['value'], best_breakpoints, right=False, labels=labels)
        remaining_df['bin'] = remaining_df['bin'].astype(str)
        
        # Create binning DataFrame
        binning_df = remaining_df.groupby(['variable', 'bin'], group_keys=False)['y'] \
            .agg([n0, n1]) \
            .reset_index() \
            .rename(columns={'n0': 'good', 'n1': 'bad'})
        
        # Check and correct empty bins
        binning_df = check_and_correct_empty_bins(remaining_df, binning_df)
        
        # Merge binning with breakpoints and sort
        binning_df = pd.merge(
            binning_df.assign(value=lambda x: [float(re.search(r"^\[(.*),(.*)\)", i).group(2)) if i != 'nan' else np.nan for i in binning_df['bin']]),
            breakpoints_df.assign(value=lambda x: x['value'].astype(float)), 
            how='left', on='value'
        ).sort_values(by="rowid").reset_index(drop=True)
        
        # Handle NaN values in breakpoints
        if breakpoints_df['value'].isnull().any():
            binning_df = binning_df.assign(bin=lambda x: [i if i != 'nan' else 'missing' for i in x['bin']]) \
                .fillna('missing') \
                .groupby(['variable', 'rowid']) \
                .agg({'bin': lambda x: '%,%'.join(x), 'good': sum, 'bad': sum}) \
                .reset_index()
    else:
        # Merge binning with breakpoints for non-numeric values
        binning_df = pd.merge(
            remaining_df,
            breakpoints_df.assign(bin=lambda x: x['bin_chr']),
            how='left', on='value'
        ).fillna('missing') \
        .groupby(['variable', 'rowid', 'bin'], group_keys=False)['y'] \
        .agg([n0, n1]) \
        .rename(columns={'n0': 'good', 'n1': 'bad'}) \
        .reset_index() \
        .drop('rowid', axis=1)
    
    return {'binning_sv': special_values_binning, 'binning': binning_df}

def pretty(low, high, num_intervals):
    '''
    Generate pretty breakpoints, similar to the R 'pretty' function.
    
    Parameters
    ----------
    low : float
        Minimum value.
    high : float
        Maximum value.
    num_intervals : int
        Number of intervals.
    
    Returns
    -------
    numpy.ndarray
        Array of breakpoints.
    '''
    # Function to generate nice numbers
    def nice_number(x):
        exp = np.floor(np.log10(abs(x)))
        f = abs(x) / 10**exp
        if f < 1.5:
            nice_f = 1.
        elif f < 3.:
            nice_f = 2.
        elif f < 7.:
            nice_f = 5.
        else:
            nice_f = 10.
        return np.sign(x) * nice_f * 10.**exp
    
    # Calculate pretty breakpoints
    interval = abs(nice_number((high - low) / (num_intervals - 1)))
    min_value = np.floor(low / interval) * interval
    max_value = np.ceil(high / interval) * interval
    return np.arange(min_value, max_value + 0.5 * interval, interval)

def woebin2_init_bin(melted_df, min_count_distribution, breaks, special_values):
    '''
    Initial binning process.

    Parameters
    ----------
    melted_df : pandas.DataFrame
        Melted DataFrame containing the values to be binned.
    min_count_distribution : float
        Minimal percentage in the fine binning process.
    breaks : list
        List of breakpoints.
    special_values : list
        List of special values to be considered separately.

    Returns
    -------
    dict
        A dictionary with initial binning and special value binning.
    '''
    
    # Extract binning for special values
    binning_result = dtm_binning_sv(melted_df, breaks, special_values)
    updated_df = binning_result['melted_df']
    special_values_binning = binning_result['binning_sv']
    
    if updated_df is None:
        return {'binning_sv': special_values_binning, 'initial_binning': None}
    
    # Initial binning for numeric variables
    if is_numeric_dtype(updated_df['value']):
        numeric_values = updated_df['value'].astype(float)
        # Compute quantiles and interquartile range
        quantiles = numeric_values.quantile([0.01, 0.25, 0.75, 0.99])
        iqr = quantiles[0.75] - quantiles[0.25]
        prob_down = 0.01 if iqr == 0 else 0.25
        prob_up = 0.99 if iqr == 0 else 0.75
        
        # Remove outliers
        filtered_values = numeric_values[(numeric_values >= quantiles[prob_down] - 3 * iqr) &
                                         (numeric_values <= quantiles[prob_up] + 3 * iqr)]
        
        # Number of initial bins
        num_bins = int(np.trunc(1 / min_count_distribution))
        unique_values_count = len(np.unique(filtered_values))
        if unique_values_count < num_bins:
            num_bins = unique_values_count
        
        # Determine initial breakpoints
        breakpoints = np.unique(filtered_values) if unique_values_count < 10 else \
                       pretty(min(filtered_values), max(filtered_values), num_bins)
        
        breakpoints = [float('-inf')] + sorted(filter(lambda x: x > np.nanmin(numeric_values) and x <= np.nanmax(numeric_values), breakpoints)) + [float('inf')]
        
        # Create initial binning DataFrame
        labels = ['[{},{})'.format(breakpoints[i], breakpoints[i+1]) for i in range(len(breakpoints) - 1)]
        updated_df['bin'] = pd.cut(updated_df['value'], breakpoints, right=False, labels=labels)
        
        # Aggregate and format initial binning DataFrame
        initial_binning_df = updated_df.groupby('bin', group_keys=False)['y'] \
            .agg([n0, n1]) \
            .reset_index() \
            .rename(columns={'n0': 'good', 'n1': 'bad'})
        
        initial_binning_df = check_empty_bins(updated_df, initial_binning_df)
        
        initial_binning_df = initial_binning_df.assign(
            variable = updated_df['variable'].values[0],
            breakpoint = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']],
            bad_probability = lambda x: x['bad'] / (x['bad'] + x['good'])
        )[['variable', 'bin', 'breakpoint', 'good', 'bad', 'bad_probability']]
    else:
        # Initial binning for non-numeric variables
        initial_binning_df = updated_df.groupby('value', group_keys=False)['y'] \
            .agg([n0, n1]) \
            .rename(columns={'n0': 'good', 'n1': 'bad'}) \
            .assign(
                variable = updated_df['variable'].values[0],
                bad_probability = lambda x: x['bad'] / (x['bad'] + x['good'])
            ) \
            .reset_index()
        
        if updated_df['value'].dtype.name not in ['category', 'bool']:
            initial_binning_df = initial_binning_df.sort_values(by='bad_probability').reset_index()
        
        initial_binning_df = initial_binning_df.assign(
            breakpoint = lambda x: x.index
        )[['variable', 'value', 'breakpoint', 'good', 'bad', 'bad_probability']] \
        .rename(columns={'value': 'bin'})
    
    # Remove breakpoints with zero good or bad counts
    while len(initial_binning_df.query('(good == 0) or (bad == 0)')) > 0:
        to_remove = initial_binning_df.assign(
            total_count = lambda x: x['good'] + x['bad']
        ).assign(
            count_lag = lambda x: x['total_count'].shift(1).fillna(len(updated_df) + 1),
            count_lead = lambda x: x['total_count'].shift(-1).fillna(len(updated_df) + 1)
        ).assign(
            merge_to_lead = lambda x: x['count_lag'] > x['count_lead']
        ).query('(good == 0) or (bad == 0)') \
        .query('total_count == total_count.min()').iloc[0]
        
        shift_period = -1 if to_remove['merge_to_lead'] else 1
        initial_binning_df = initial_binning_df.assign(
            new_breakpoint = lambda x: x['breakpoint'].shift(shift_period)
        ).assign(
            breakpoint = lambda x: np.where(x['breakpoint'] == to_remove['breakpoint'], x['new_breakpoint'], x['breakpoint'])
        )
        
        initial_binning_df = initial_binning_df.groupby('breakpoint', group_keys=False).agg({
            'variable': lambda x: np.unique(x)[0],
            'bin': lambda x: '%,%'.join(x),
            'good': sum,
            'bad': sum
        }).assign(
            bad_probability = lambda x: x['bad'] / (x['good'] + x['bad'])
        ).reset_index()
    
    # Format initial binning DataFrame
    if is_numeric_dtype(updated_df['value']):
        initial_binning_df = initial_binning_df \
            .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']]) \
            .assign(breakpoint = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    
    return {'binning_sv': special_values_binning, 'initial_binning': initial_binning_df}


def woebin2_tree_add_1brkp(melted_df, initial_binning, count_distribution_limit, best_breaks=None):
    '''
    Add one best breakpoint to the provided best_breaks.

    Parameters
    ----------
    melted_df : pandas.DataFrame
        Melted DataFrame containing the values to be binned.
    initial_binning : pandas.DataFrame
        DataFrame containing the initial binning information.
    count_distribution_limit : float
        Minimum count distribution limit for breakpoints.
    best_breaks : list, optional
        List of current best breakpoints. If None, only one breakpoint will be added.

    Returns
    -------
    pandas.DataFrame
        DataFrame with updated breakpoints.
    '''
    
    def total_iv_all_breaks(initial_binning, best_breaks, total_rows):
        # Determine possible breakpoints
        available_breaks = set(initial_binning['breakpoint']).difference(set(map(float, ['-inf', 'inf'])))
        if best_breaks is not None:
            available_breaks = available_breaks.difference(set(best_breaks))
        available_breaks = sorted(available_breaks)
        
        # Calculate IV for all possible breakpoints
        all_breaks_df = initial_binning.copy(deep=True)
        for break_point in available_breaks:
            new_breaks = [float('-inf')] + sorted(best_breaks + [break_point] if best_breaks is not None else [break_point]) + [float('inf')]
            labels = ['[{},{})'.format(new_breaks[i], new_breaks[i+1]) for i in range(len(new_breaks) - 1)]
            all_breaks_df[f'bestbin{break_point}'] = pd.cut(all_breaks_df['breakpoint'], new_breaks, right=False, labels=labels)
        
        melted_df_all_breaks = pd.melt(
            all_breaks_df, id_vars=["variable", "good", "bad"], var_name='bestbin', 
            value_vars=[f'bestbin{break_point}' for break_point in available_breaks]
        ).groupby(['variable', 'bestbin', 'value'], group_keys=False) \
         .agg({'good': sum, 'bad': sum}).reset_index() \
         .assign(total_count=lambda x: x['good'] + x['bad'])
        
        melted_df_all_breaks['count_distribution'] = melted_df_all_breaks.groupby(['variable', 'bestbin'], group_keys=False) \
            ['total_count'].apply(lambda x: x / total_rows).reset_index(drop=True)
        melted_df_all_breaks['min_count_distribution'] = melted_df_all_breaks.groupby(['variable', 'bestbin'], group_keys=False) \
            ['count_distribution'].transform(lambda x: min(x))
        
        melted_df_all_breaks = melted_df_all_breaks \
            .assign(bestbin=lambda x: [float(re.sub('^bestbin', '', i)) for i in x['bestbin']]) \
            .groupby(['variable', 'bestbin', 'min_count_distribution'], group_keys=False) \
            .apply(lambda x: iv_01(x['good'], x['bad'])).reset_index(name='total_iv')
        
        return melted_df_all_breaks
    
    def binning_add_1_best_break(initial_binning, best_breaks):
        if best_breaks is None:
            best_breaks_ext = [float('-inf'), float('inf')]
        else:
            if not is_numeric_dtype(melted_df['value']):
                best_breaks = [i for i in best_breaks if int(i) != min(initial_binning['breakpoint'])]
            best_breaks_ext = [float('-inf')] + sorted(best_breaks) + [float('inf')]
        
        labels = ['[{},{})'.format(best_breaks_ext[i], best_breaks_ext[i+1]) for i in range(len(best_breaks_ext) - 1)]
        updated_binning_df = initial_binning.assign(
            bestbin=lambda x: pd.cut(x['breakpoint'], best_breaks_ext, right=False, labels=labels)
        )
        
        if is_numeric_dtype(melted_df['value']):
            updated_binning_df = updated_binning_df.groupby(['variable', 'bestbin'], group_keys=False) \
                .agg({'good': sum, 'bad': sum}) \
                .reset_index() \
                .assign(bin=lambda x: x['bestbin']) \
                [['bestbin', 'variable', 'bin', 'good', 'bad']]
        else:
            updated_binning_df = updated_binning_df.groupby(['variable', 'bestbin'], group_keys=False) \
                .agg({'good': sum, 'bad': sum, 'bin': lambda x: '%,%'.join(x)}) \
                .reset_index() \
                [['bestbin', 'variable', 'bin', 'good', 'bad']]
        
        updated_binning_df['total_iv'] = iv_01(updated_binning_df['good'], updated_binning_df['bad'])
        updated_binning_df['best_breakpoint'] = [float(re.match("^\[(.*),.+", i).group(1)) for i in updated_binning_df['bestbin']]
        
        return updated_binning_df
    
    total_rows = len(melted_df)
    # Calculate IV for all potential best breakpoints
    iv_for_all_breaks = total_iv_all_breaks(initial_binning, best_breaks, total_rows)
    
    # Select the best breakpoint that maximizes IV and meets the count distribution limit
    best_break = iv_for_all_breaks.loc[lambda x: x['min_count_distribution'] >= count_distribution_limit]
    if len(best_break.index) > 0:
        best_break = best_break.loc[lambda x: x['total_iv'] == max(x['total_iv'])]
        best_break = best_break['bestbin'].tolist()[0]
    else:
        best_break = None
    
    # Add the best breakpoint to the list and update binning
    if best_break is not None:
        best_breaks = best_breaks + [best_break] if best_breaks is not None else [best_break]
    
    updated_binning = binning_add_1_best_break(initial_binning, best_breaks)
    
    return updated_binning
    
def woebin2_tree(melted_df, init_count_distr=0.02, count_distr_limit=0.05, 
                 stop_limit=0.1, bin_num_limit=8, breaks=None, special_values=None):
    '''
    Binning using a tree-like method.

    Parameters
    ----------
    melted_df : pandas.DataFrame
        Melted DataFrame containing the values to be binned.
    init_count_distr : float
        Minimal percentage in the initial binning process.
    count_distr_limit : float
        Minimal percentage in the fine binning process.
    stop_limit : float
        Stop limit for the Information Value (IV) change ratio.
    bin_num_limit : int
        Maximum number of bins.
    breaks : list, optional
        List of breakpoints to be used in binning.
    special_values : list, optional
        List of special values to be considered separately.

    Returns
    -------
    dict
        A dictionary with initial binning and special value binning.
    '''
    
    # Initial binning
    bin_list = woebin2_init_bin(melted_df, init_count_distr=init_count_distr, breaks=breaks, special_values=special_values)
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    
    if initial_binning is None:
        return {'binning_sv': binning_sv, 'binning': None}
    if len(initial_binning.index) == 1:
        return {'binning_sv': binning_sv, 'binning': initial_binning}
    
    # Initialize parameters
    len_brks = len(initial_binning.index)
    best_breaks = None
    IV_t1 = IV_t2 = 1e-10
    IV_change = 1  # IV gain ratio
    step_num = 1
    
    # Best breaks from three to n+1 bins
    binning_tree = None
    while (IV_change >= stop_limit) and (step_num + 1 <= min([bin_num_limit, len_brks])):
        binning_tree = woebin2_tree_add_1brkp(melted_df, initial_binning, count_distr_limit, best_breaks)
        best_breaks = binning_tree.loc[lambda x: x['best_break'] != float('-inf'), 'best_break'].tolist()
        IV_t2 = binning_tree['total_iv'].tolist()[0]
        IV_change = IV_t2 / IV_t1 - 1  # Ratio gain
        IV_t1 = IV_t2
        step_num += 1
    
    if binning_tree is None:
        binning_tree = initial_binning
    
    # Return results
    return {'binning_sv': binning_sv, 'binning': binning_tree}


def woebin2_chimerge(melted_df, init_count_distr=0.02, count_distr_limit=0.05, 
                     stop_limit=0.1, bin_num_limit=8, breaks=None, special_values=None):
    '''
    Binning using the Chimerge method.

    Parameters
    ----------
    melted_df : pandas.DataFrame
        Melted DataFrame containing the values to be binned.
    init_count_distr : float
        Minimal percentage in the initial binning process.
    count_distr_limit : float
        Minimal percentage in the fine binning process.
    stop_limit : float
        Stop limit for the chi-square statistic.
    bin_num_limit : int
        Maximum number of bins.
    breaks : list, optional
        List of breakpoints to be used in binning.
    special_values : list, optional
        List of special values to be considered separately.

    Returns
    -------
    dict
        A dictionary with initial binning and special value binning.
    '''
    
    def add_chisq(initial_binning):
        '''
        Function to create a chi-square column in the initial binning DataFrame.

        Parameters
        ----------
        initial_binning : pandas.DataFrame
            DataFrame with initial binning information.

        Returns
        -------
        pandas.DataFrame
            DataFrame with chi-square values added.
        '''
        chisq_df = pd.melt(initial_binning, 
            id_vars=["breakpoint", "variable", "bin"], value_vars=["good", "bad"],
            var_name='good_bad', value_name='a') \
        .sort_values(by=['good_bad', 'breakpoint']).reset_index(drop=True)
        
        chisq_df['a_lag'] = chisq_df.groupby('good_bad', group_keys=False)['a'].apply(lambda x: x.shift(1))
        chisq_df['a_rowsum'] = chisq_df.groupby('breakpoint', group_keys=False)['a'].transform(lambda x: sum(x))
        chisq_df['a_lag_rowsum'] = chisq_df.groupby('breakpoint', group_keys=False)['a_lag'].transform(lambda x: sum(x))
        
        chisq_df = pd.merge(
            chisq_df.assign(a_colsum=lambda df: df.a + df.a_lag),
            chisq_df.groupby('breakpoint', group_keys=False).apply(lambda df: sum(df.a + df.a_lag)).reset_index(name='a_sum')
        ).assign(
            e=lambda df: df.a_rowsum * df.a_colsum / df.a_sum,
            e_lag=lambda df: df.a_lag_rowsum * df.a_colsum / df.a_sum
        ).assign(
            ae=lambda df: (df.a - df.e) ** 2 / df.e + (df.a_lag - df.e_lag) ** 2 / df.e_lag
        ).groupby('breakpoint').apply(lambda x: sum(x.ae)).reset_index(name='chisq')
        
        return pd.merge(initial_binning.assign(count=lambda x: x['good'] + x['bad']), chisq_df, how='left')
    
    # Initial binning
    bin_list = woebin2_init_bin(melted_df, init_count_distr=init_count_distr, breaks=breaks, special_values=special_values)
    initial_binning = bin_list['initial_binning']
    binning_sv = bin_list['binning_sv']
    
    if len(initial_binning.index) == 1:
        return {'binning_sv': binning_sv, 'binning': initial_binning}
    
    # DataFrame rows
    total_rows = len(melted_df.index)
    
    # Chi-square limit
    chisq_limit = chdtri(1, stop_limit)
    
    # Binning with chi-square column
    binning_chisq = add_chisq(initial_binning)
    
    # Parameters
    min_chisq = binning_chisq.chisq.min()
    min_count_distr = min(binning_chisq['count'] / total_rows)
    bin_nrow = len(binning_chisq.index)
    
    while min_chisq < chisq_limit or min_count_distr < count_distr_limit or bin_nrow > bin_num_limit:
        if min_chisq < chisq_limit:
            remove_brkp = binning_chisq.assign(merge_to_lead=False).sort_values(by=['chisq', 'count']).iloc[0]
        elif min_count_distr < count_distr_limit:
            remove_brkp = binning_chisq.assign(
                count_distr=lambda x: x['count'] / sum(x['count']),
                chisq_lead=lambda x: x['chisq'].shift(-1).fillna(float('inf'))
            ).assign(merge_to_lead=lambda x: x['chisq'] > x['chisq_lead'])
            remove_brkp.loc[np.isnan(remove_brkp['chisq']), 'merge_to_lead'] = True
            remove_brkp = remove_brkp.sort_values(by=['count_distr']).iloc[0]
        elif bin_nrow > bin_num_limit:
            remove_brkp = binning_chisq.assign(merge_to_lead=False).sort_values(by=['chisq', 'count']).iloc[0]
        else:
            break
        
        shift_period = -1 if remove_brkp['merge_to_lead'] else 1
        binning_chisq = binning_chisq.assign(
            new_breakpoint=lambda x: x['breakpoint'].shift(shift_period)
        ).assign(
            breakpoint=lambda x: np.where(x['breakpoint'] == remove_brkp['breakpoint'], x['new_breakpoint'], x['breakpoint'])
        )
        
        # Group by breakpoint
        binning_chisq = binning_chisq.groupby('breakpoint', group_keys=False).agg({
            'variable': lambda x: np.unique(x),
            'bin': lambda x: '%,%'.join(x),
            'good': sum,
            'bad': sum
        }).assign(bad_probability=lambda x: x['bad'] / (x['good'] + x['bad'])) \
        .reset_index()
        
        # Update
        binning_chisq = add_chisq(binning_chisq)
        
        bin_nrow = len(binning_chisq.index)
        if bin_nrow == 1:
            break
        
        min_chisq = binning_chisq.chisq.min()
        min_count_distr = min(binning_chisq['count'] / total_rows)
    
    # Format initial binning
    if pd.api.types.is_numeric_dtype(melted_df['value']):
        binning_chisq = binning_chisq \
        .assign(bin=lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']]) \
        .assign(breakpoint=lambda x: [float(re.match(r'^\[(.*),.+', i).group(1)) for i in x['bin']])
    
    # Return results
    return {'binning_sv': binning_sv, 'binning': binning_chisq}
     
     
def binning_format(binning):
    '''
    Format the binning DataFrame for output.

    Parameters
    ----------
    binning : pandas.DataFrame
        DataFrame with columns of 'variable', 'bin', 'good', 'bad'.

    Returns
    -------
    pandas.DataFrame
        Formatted DataFrame with additional columns:
        'variable', 'bin', 'count', 'count_distr', 'good', 'bad', 
        'badprob', 'woe', 'bin_iv', 'total_iv', 'breaks', 'is_special_values'.
    '''
    # Compute additional metrics
    binning['count'] = binning['good'] + binning['bad']
    binning['count_distr'] = binning['count'] / binning['count'].sum()
    binning['badprob'] = binning['bad'] / binning['count']
    binning['woe'] = woe_01(binning['good'], binning['bad'])
    binning['bin_iv'] = miv_01(binning['good'], binning['bad'])
    binning['total_iv'] = binning['bin_iv'].sum()

    # Extract breaks from bin if applicable
    binning['breaks'] = binning['bin']
    if any('[' in str(i) for i in binning['bin']):
        def extract_breaks(x):
            match = re.match(r"^\[(.*), *(.*)\)((%,%missing)*)", x)
            return x if match is None else match.group(2) + match.group(3)
        binning['breaks'] = [extract_breaks(i) for i in binning['bin']]
    
    # Special values flag
    binning['is_special_values'] = binning.get('is_sv', np.nan)
    
    # Return formatted DataFrame
    return binning[['variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 'bin_iv', 'total_iv', 'breaks', 'is_special_values']]

def woebin2(dtm, breaks=None, special_values=None, 
            init_count_distr=0.02, count_distr_limit=0.05, 
            stop_limit=0.1, bin_num_limit=8, method="tree"):
    '''
    Perform WOE binning for two columns (one x and one y) DataFrame.

    Parameters
    ----------
    dtm : pandas.DataFrame
        DataFrame with columns to be binned.
    breaks : list, optional
        List of breakpoints for binning.
    special_values : list, optional
        List of special values.
    init_count_distr : float, default=0.02
        Minimal percentage for initial binning.
    count_distr_limit : float, default=0.05
        Minimal percentage for final binning.
    stop_limit : float, default=0.1
        Stop limit for IV or chi-square.
    bin_num_limit : int, default=8
        Maximum number of bins.
    method : str, default="tree"
        Binning method, either "tree" or "chimerge".

    Returns
    -------
    pandas.DataFrame
        DataFrame with binned data, formatted by `binning_format`.
    '''
    # Perform binning based on method
    if breaks is not None:
        bin_list = woebin2_breaks(dtm=dtm, breaks=breaks, special_values=special_values)
    else:
        if stop_limit == 'N':
            bin_list = woebin2_init_bin(dtm, init_count_distr=init_count_distr, breaks=breaks, special_values=special_values)
        else:
            if method == 'tree':
                bin_list = woebin2_tree(
                    dtm, init_count_distr=init_count_distr, count_distr_limit=count_distr_limit, 
                    stop_limit=stop_limit, bin_num_limit=bin_num_limit, breaks=breaks, special_values=special_values)
            elif method == "chimerge":
                bin_list = woebin2_chimerge(
                    dtm, init_count_distr=init_count_distr, count_distr_limit=count_distr_limit, 
                    stop_limit=stop_limit, bin_num_limit=bin_num_limit, breaks=breaks, special_values=special_values)

    # Combine and format binning results
    binning = pd.concat(bin_list, keys=bin_list.keys()).reset_index() \
              .assign(is_special_values=lambda x: x.level_0 == 'binning_sv')
    
    return binning_format(binning)

def bins_to_breaks(bins, dt, to_string=False, save_string=None):
    '''
    Convert binned data to breakpoints.

    Parameters
    ----------
    bins : pandas.DataFrame or dict
        DataFrame or dictionary of binning results.
    dt : pandas.DataFrame
        DataFrame with original data.
    to_string : bool, default=False
        Whether to convert the result to a string format.
    save_string : str, optional
        If provided, save the string representation to a file with this name.

    Returns
    -------
    dict or None
        Dictionary of breakpoints or None if `to_string=True`.
    '''
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)

    # Identify variables and their types
    unique_vars = bins['variable'].unique()
    var_types = pd.DataFrame({
        'variable': unique_vars,
        'is_numeric': [is_numeric_dtype(dt[var]) for var in unique_vars]
    })
    
    # Extract and format breaks
    breaks_list = bins[~bins['breaks'].isin(["-inf", "inf", "missing"]) & ~bins['is_special_values']]
    breaks_list = pd.merge(breaks_list[['variable', 'breaks']], var_types, how='left', on='variable')
    breaks_list.loc[~breaks_list['is_numeric'], 'breaks'] = '\'' + breaks_list.loc[~breaks_list['is_numeric'], 'breaks'] + '\''
    breaks_list = breaks_list.groupby('variable', group_keys=False)['breaks'].agg(lambda x: ','.join(x))
    
    if to_string:
        breaks_str = "breaks_list={\n" + ', \n'.join(f"'{var}': [{brk}]" for var, brk in breaks_list.items()) + "}"
        if save_string is not None:
            filename = f"{save_string}_{time.strftime('%Y%m%d_%H%M%S')}.py"
            with open(filename, 'w') as f:
                f.write(breaks_str)
            print(f'[INFO] The breaks_list is saved as {filename}')
            return 
        return breaks_str
    
    return breaks_list

def woebin(dt, y, x=None, 
           var_skip=None, breaks_list=None, special_values=None, 
           stop_limit=0.1, count_distr_limit=0.05, bin_num_limit=8, 
           positive="bad|1", no_cores=None, print_step=0, method="tree",
           ignore_const_cols=True, ignore_datetime_cols=True, 
           check_cate_num=True, replace_blank=True, 
           save_breaks_list=None, **kwargs):
    '''
    Perform WOE binning on specified variables in a DataFrame.

    Parameters
    ----------
    dt : pandas.DataFrame
        DataFrame containing the data.
    y : str
        Name of the response variable.
    x : list of str, optional
        Names of the predictor variables. If None, all columns except y are used.
    var_skip : list of str, optional
        Variables to skip during binning.
    breaks_list : dict, optional
        Custom breakpoints for binning.
    special_values : dict, optional
        Special values to handle separately.
    stop_limit : float, default=0.1
        Stop limit for binning. Binning stops when IV gain ratio or chi-square is below this value.
    count_distr_limit : float, default=0.05
        Minimum percentage of final bin class number over total.
    bin_num_limit : int, default=8
        Maximum number of bins.
    positive : str, default="bad|1"
        Indicates the positive class. Change to "good" or "0" for other conventions.
    no_cores : int, optional
        Number of CPU cores for parallel computation. Defaults to None.
    print_step : int, default=0
        Print progress information every `print_step` iterations.
    method : str, default="tree"
        Binning method. Options are "tree" or "chimerge".
    ignore_const_cols : bool, default=True
        Whether to ignore constant columns.
    ignore_datetime_cols : bool, default=True
        Whether to ignore datetime columns.
    check_cate_num : bool, default=True
        Whether to check for too many unique values in categorical columns.
    replace_blank : bool, default=True
        Whether to replace blank values with NaN.
    save_breaks_list : str, optional
        Filename to save the breaks_list. If None, do not save.

    Returns
    -------
    dict
        Dictionary with the results of binning for each variable.
    '''
    # Start time for performance tracking
    start_time = time.time()
    
    # Handle default values for kwargs
    init_count_distr = kwargs.get('min_perc_fine_bin', 0.02)
    count_distr_limit = kwargs.get('min_perc_coarse_bin', 0.05)
    bin_num_limit = kwargs.get('max_num_bin', 8)
    
    if kwargs.get('print_info', True):
        print('[INFO] Creating WOE binning...')
    
    # Data preparation
    dt = dt.copy(deep=True)
    y = [y] if isinstance(y, str) else y
    x = [x] if isinstance(x, str) and x is not None else x
    if x is not None:
        dt = dt[y + x]

    # Validate and preprocess data
    dt = check_y(dt, y, positive)
    if ignore_const_cols: dt = check_const_cols(dt)
    if ignore_datetime_cols: dt = check_datetime_cols(dt)
    if check_cate_num: check_cateCols_uniqueValues(dt, var_skip)
    if replace_blank: dt = rep_blank_na(dt)
    
    # Determine variables to bin
    xs = x_variable(dt, y, x, var_skip)
    xs_len = len(xs)
    
    # Validate parameters
    validate_params(stop_limit, init_count_distr, count_distr_limit, bin_num_limit, method)
    
    # Determine number of cores for parallel processing
    if no_cores is None:
        no_cores = determine_no_cores(xs_len)
    if platform.system() == 'Windows':
        no_cores = 1
    
    # Bin variables
    bins = perform_binning(dt, y[0], xs, breaks_list, special_values, 
                           init_count_distr, count_distr_limit, stop_limit, 
                           bin_num_limit, method, no_cores, print_step)
    
    # Track runtime
    runtime = time.time() - start_time
    if runtime >= 10 and kwargs.get('print_info', True):
        print(f'Binning on {dt.shape[0]} rows and {dt.shape[1]} columns took {time.strftime("%H:%M:%S", time.gmtime(runtime))}')
    
    # Save breaks list if specified
    if save_breaks_list is not None:
        bins_to_breaks(bins, dt, to_string=True, save_string=save_breaks_list)
    
    return bins

def validate_params(stop_limit, init_count_distr, count_distr_limit, bin_num_limit, method):
    '''
    Validate and correct the parameters for binning.

    Parameters
    ----------
    stop_limit : float
        Stop limit for binning.
    init_count_distr : float
        Minimum percentage for initial binning.
    count_distr_limit : float
        Minimum percentage for final binning.
    bin_num_limit : int
        Maximum number of bins.
    method : str
        Binning method.
    '''
    if not (0 <= stop_limit <= 0.5):
        warnings.warn("stop_limit should be between 0 and 0.5. Setting to default (0.1).")
        stop_limit = 0.1
    if not (0.01 <= init_count_distr <= 0.2):
        warnings.warn("init_count_distr should be between 0.01 and 0.2. Setting to default (0.02).")
        init_count_distr = 0.02
    if not (0.01 <= count_distr_limit <= 0.2):
        warnings.warn("count_distr_limit should be between 0.01 and 0.2. Setting to default (0.05).")
        count_distr_limit = 0.05
    if not isinstance(bin_num_limit, int):
        warnings.warn("bin_num_limit should be an integer. Setting to default (8).")
        bin_num_limit = 8
    if method not in ["tree", "chimerge"]:
        warnings.warn("method should be 'tree' or 'chimerge'. Setting to default ('tree').")
        method = "tree"

def determine_no_cores(xs_len):
    '''
    Determine the number of cores to use for parallel processing.

    Parameters
    ----------
    xs_len : int
        Number of variables to bin.

    Returns
    -------
    int
        Number of cores to use.
    '''
    if xs_len < 10:
        return 1
    return max(1, int(np.ceil(mp.cpu_count() * 0.9)))

def perform_binning(dt, y, xs, breaks_list, special_values, 
                    init_count_distr, count_distr_limit, stop_limit, 
                    bin_num_limit, method, no_cores, print_step):
    '''
    Perform binning on each variable and handle parallel processing.

    Parameters
    ----------
    dt : pandas.DataFrame
        DataFrame containing the data.
    y : str
        Name of the response variable.
    xs : list of str
        List of predictor variables.
    breaks_list : dict
        Custom breakpoints for binning.
    special_values : dict
        Special values to handle separately.
    init_count_distr : float
        Minimum percentage for initial binning.
    count_distr_limit : float
        Minimum percentage for final binning.
    stop_limit : float
        Stop limit for binning.
    bin_num_limit : int
        Maximum number of bins.
    method : str
        Binning method.
    no_cores : int
        Number of CPU cores for parallel processing.
    print_step : int
        Print progress information every `print_step` iterations.

    Returns
    -------
    dict
        Dictionary with the results of binning for each variable.
    '''
    if no_cores == 1:
        bins = {}
        for i, x in enumerate(xs):
            if print_step > 0 and (i + 1) % print_step == 0:
                print(f'{i + 1}/{len(xs)}: {x}', flush=True)
            bins[x] = woebin2(
                dtm=pd.DataFrame({'y': dt[y], 'variable': x, 'value': dt[x]}),
                breaks=breaks_list.get(x),
                spl_val=special_values.get(x),
                init_count_distr=init_count_distr,
                count_distr_limit=count_distr_limit,
                stop_limit=stop_limit,
                bin_num_limit=bin_num_limit,
                method=method
            )
    else:
        with mp.Pool(processes=no_cores) as pool:
            args = [(pd.DataFrame({'y': dt[y], 'variable': x, 'value': dt[x]}),
                     breaks_list.get(x),
                     special_values.get(x),
                     init_count_distr,
                     count_distr_limit,
                     stop_limit,
                     bin_num_limit,
                     method) for x in xs]
            bins = dict(zip(xs, pool.starmap(woebin2, args)))
    return bins



def woepoints_ply1(dtx, binx, x_i, woe_points):
    '''
    Transform original values into WOE or points for one variable.

    Parameters
    ----------
    dtx : pandas.DataFrame
        DataFrame containing the original values.
    binx : pandas.DataFrame
        DataFrame containing the binning information.
    x_i : str
        The name of the variable to transform.
    woe_points : str
        Indicates whether to use "woe" or "points" for transformation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with transformed values.
    '''
    # Split 'bin' column into separate rows
    binx = pd.merge(
        binx[['bin']].assign(v1=binx['bin'].str.split('%,%')).explode('v1'),
        binx[['bin', woe_points]],
        how='left', on='bin'
    ).rename(columns={'v1': 'V1', woe_points: 'V2'})

    # Handle numeric variables
    if is_numeric_dtype(dtx[x_i]):
        is_sv = pd.Series(~dtx[x_i].astype(str).str.contains(r'\[', na=False))
        binx_sv = binx[is_sv]
        binx_other = binx[~is_sv]

        # Create bin column for non-special values
        breaks_binx_other = np.unique(
            list(map(float, ['-inf'] + [re.match(r'.*\[(.*),.+\).*', str(i)).group(1) for i in binx_other['bin']] + ['inf']))
        )
        labels = [f'[{breaks_binx_other[i]},{breaks_binx_other[i+1]})' for i in range(len(breaks_binx_other) - 1)]

        dtx = dtx.assign(
            xi_bin=pd.cut(dtx[x_i], breaks_binx_other, right=False, labels=labels)
        )
        dtx['xi_bin'] = dtx['xi_bin'].astype(str).replace('nan', 'missing')
        
        # Update bins for special values
        try:
            mask = dtx[x_i].astype('int').isin(binx_sv['V1'].astype('int'))
        except:
            mask = dtx[x_i].isin(binx_sv['V1'])
        
        dtx.loc[mask, 'xi_bin'] = dtx.loc[mask, x_i].astype(str)
        dtx = dtx[['xi_bin']].rename(columns={'xi_bin': x_i})
    else:
        # Convert non-numeric variables to string and handle missing values
        if not is_string_dtype(dtx[x_i]):
            dtx[x_i] = dtx[x_i].astype(str).replace('nan', 'missing')

        dtx = dtx.replace(np.nan, 'missing')

    # Merge with binx to get WOE or points values
    binx.columns = ['bin', x_i, '_'.join([x_i, woe_points])]
    dtx = pd.merge(dtx, binx, how='left', on=x_i)
    
    return dtx.set_index(dtx.index)[['_'.join([x_i, woe_points])]]

def woebin_ply(dt, bins, no_cores=None, print_step=0, replace_blank=True, **kwargs):
    '''
    Apply WOE transformation to a DataFrame based on binning information.

    Parameters
    ----------
    dt : pandas.DataFrame
        DataFrame containing the original data.
    bins : dict or pandas.DataFrame
        Binning information. If a dict, should be a dict of DataFrames.
    no_cores : int, optional
        Number of CPU cores for parallel computation. If None, it will be automatically set.
    print_step : int, default=0
        Print progress information every `print_step` iterations.
    replace_blank : bool, default=True
        Whether to replace blank values with 'missing'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with WOE values for each variable.
    '''
    # Start time for performance tracking
    start_time = time.time()
    
    # Print info
    print_info = kwargs.get('print_info', True)
    if print_info: print('[INFO] Converting into WOE values ...')
    
    # Replace blank values with 'missing'
    if replace_blank: dt = rep_blank_na(dt)
    
    # Process bins
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    
    xs_bin = bins['variable'].unique()
    xs_dt = set(dt.columns)
    xs = list(xs_bin.intersection(xs_dt))
    xs_len = len(xs)
    dat = dt[list(set(dt.columns) - set(xs))]
    
    # Determine number of cores for parallel processing
    if no_cores is None or no_cores < 1:
        no_cores = max(1, min(mp.cpu_count() - 1, int(np.ceil(xs_len / 5))))
    if platform.system() == 'Windows': 
        no_cores = 1
    
    # Define the function for parallel processing
    def process_variable(x_i):
        binx = bins[bins['variable'] == x_i].reset_index()
        dtx = dt[[x_i]]
        return woepoints_ply1(dtx, binx, x_i, woe_points="woe")
    
    # Apply WOE transformation
    if no_cores == 1:
        for i, x_i in enumerate(xs):
            if print_step > 0 and (i + 1) % print_step == 0:
                print(f'{i + 1}/{xs_len}: {x_i}', flush=True)
            dat = pd.concat([dat, process_variable(x_i)], axis=1)
    else:
        with mp.Pool(processes=no_cores) as pool:
            results = pool.map(process_variable, xs)
        dat = pd.concat([dat] + results, axis=1)
    
    # Runtime information
    run_time = time.time() - start_time
    if run_time >= 10 and print_info:
        print(f'WOE transformation on {dt.shape[0]} rows and {xs_len} columns took {time.strftime("%H:%M:%S", time.gmtime(run_time))}')
    
    return dat
    


# required in woebin_plot
#' @import data.table ggplot2
def plot_bin(binx, title, show_iv):
    '''
    plot binning of one variable
    
    Params
    ------
    binx:
    title:
    show_iv:
    
    Returns
    ------
    matplotlib fig object
    '''
    # y_right_max
    y_right_max = np.ceil(binx['badprob'].max()*10)
    if y_right_max % 2 == 1: y_right_max=y_right_max+1
    if y_right_max - binx['badprob'].max()*10 <= 0.3: y_right_max = y_right_max+2
    y_right_max = y_right_max/10
    if y_right_max>1 or y_right_max<=0 or y_right_max is np.nan or y_right_max is None: y_right_max=1
    ## y_left_max
    y_left_max = np.ceil(binx['count_distr'].max()*10)/10
    if y_left_max>1 or y_left_max<=0 or y_left_max is np.nan or y_left_max is None: y_left_max=1
    # title
    title_string = binx.loc[0,'variable']+"  (iv:"+str(round(binx.loc[0,'total_iv'],4))+")" if show_iv else binx.loc[0,'variable']
    title_string = title+'-'+title_string if title is not None else title_string
    # param
    ind = np.arange(len(binx.index))    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    ###### plot ###### 
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1
    p1 = ax1.bar(ind, binx['good_distr'], width, color=(24/254, 192/254, 196/254))
    p2 = ax1.bar(ind, binx['bad_distr'], width, bottom=binx['good_distr'], color=(246/254, 115/254, 109/254))
    for i in ind:
        ax1.text(i, binx.loc[i,'count_distr']*1.02, str(round(binx.loc[i,'count_distr']*100,1))+'%, '+str(binx.loc[i,'count']), ha='center')
    # ax2
    ax2.plot(ind, binx['badprob'], marker='o', color='blue')
    for i in ind:
        ax2.text(i, binx.loc[i,'badprob']*1.02, str(round(binx.loc[i,'badprob']*100,1))+'%', color='blue', ha='center')
    # settings
    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Bad probability', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max+0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max+0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')
    plt.xticks(ind, binx['bin'])
    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')
    # show plot 
    # plt.show()
    return fig


def woebin_plot(bins, x=None, title=None, show_iv=True):
    '''
    WOE Binning Visualization
    ------
    `woebin_plot` create plots of count distribution and bad probability 
    for each bin. The binning informations are generates by `woebin`.
    
    Params
    ------
    bins: A list or data frame. Binning information generated by `woebin`.
    x: Name of x variables. Default is None. If x is None, then all 
      variables except y are counted as x variables.
    title: String added to the plot title. Default is None.
    show_iv: Logical. Default is True, which means show information value 
      in the plot title.
    
    Returns
    ------
    dict
        a dict of matplotlib figure objests
        
    Examples
    ------
    import scorecardpy as sc
    import matplotlib.pyplot as plt
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    dt1 = dat[["creditability", "credit.amount"]]
    
    bins1 = sc.woebin(dt1, y="creditability")
    p1 = sc.woebin_plot(bins1)
    plt.show(p1)
    
    # Example II
    bins = sc.woebin(dat, y="creditability")
    plotlist = sc.woebin_plot(bins)
    
    # # save binning plot
    # for key,i in plotlist.items():
    #     plt.show(i)
    #     plt.savefig(str(key)+'.png')
    '''
    xs = x
    # bins concat 
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # good bad distr
    def gb_distr(binx):
        binx['good_distr'] = binx['good']/sum(binx['count'])
        binx['bad_distr'] = binx['bad']/sum(binx['count'])
        return binx
    bins = bins.groupby('variable', group_keys=False).apply(gb_distr)
    # x variable names
    if xs is None: xs = bins['variable'].unique()
    # plot export
    plotlist = {}
    for i in xs:
        binx = bins[bins['variable'] == i].reset_index(drop=True)
        plotlist[i] = plot_bin(binx, title, show_iv)
    return plotlist 



# print basic information in woebin_adj
def woebin_adj_print_basic_info(i, xs, bins, dt, bins_breakslist):
    '''
    print basic information of woebinnig in adjusting process
    
    Params
    ------
    
    Returns
    ------
    
    '''
    x_i = xs[i-1]
    xs_len = len(xs)
    binx = bins.loc[bins['variable']==x_i]
    print("--------", str(i)+"/"+str(xs_len), x_i, "--------")
    # print(">>> dt["+x_i+"].dtypes: ")
    # print(str(dt[x_i].dtypes), '\n')
    # 
    print(">>> dt["+x_i+"].describe(): ")
    print(dt[x_i].describe(), '\n')
    
    if len(dt[x_i].unique()) < 10 or not is_numeric_dtype(dt[x_i]):
        print(">>> dt["+x_i+"].value_counts(): ")
        print(dt[x_i].value_counts(), '\n')
    else:
        dt[x_i].hist()
        plt.title(x_i)
        plt.show()
        
    ## current breaks
    print(">>> Current breaks:")
    print(bins_breakslist[x_i], '\n')
    ## woebin plotting
    plt.show(woebin_plot(binx)[x_i])
    
    
# plot adjusted binning in woebin_adj
def woebin_adj_break_plot(dt, y, x_i, breaks, stop_limit, sv_i, method):
    '''
    update breaks and provies a binning plot
    
    Params
    ------
    
    Returns
    ------
    
    '''
    if breaks == '':
        breaks = None
    breaks_list = None if breaks is None else {x_i: eval('['+breaks+']')}
    special_values = None if sv_i is None else {x_i: sv_i}
    # binx update
    bins_adj = woebin(dt[[x_i,y]], y, breaks_list=breaks_list, special_values=special_values, stop_limit = stop_limit, method=method)
    
    ## print adjust breaks
    breaks_bin = set(bins_adj[x_i]['breaks']) - set(["-inf","inf","missing"])
    breaks_bin = ', '.join(breaks_bin) if is_numeric_dtype(dt[x_i]) else ', '.join(['\''+ i+'\'' for i in breaks_bin])
    print(">>> Current breaks:")
    print(breaks_bin, '\n')
    # print bin_adj
    plt.show(woebin_plot(bins_adj))
    # return breaks 
    if breaks == '' or breaks is None: breaks = breaks_bin
    return breaks
    
    
def woebin_adj(dt, y, bins, adj_all_var=False, special_values=None, method="tree", save_breaks_list=None, count_distr_limit=0.05):
    '''
    WOE Binning Adjustment
    ------
    `woebin_adj` interactively adjust the binning breaks.
    
    Params
    ------
    dt: A data frame.
    y: Name of y variable.
    bins: A list or data frame. Binning information generated from woebin.
    adj_all_var: Logical, whether to show monotonic woe variables. Default
      is True
    special_values: the values specified in special_values will in separate 
      bins. Default is None.
    method: optimal binning method, it should be "tree" or "chimerge". 
      Default is "tree".
    save_breaks_list: The file name to save breaks_list. Default is None.
    count_distr_limit: The minimum percentage of final binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.05.

    
    Returns
    ------
    dict
        dictionary of breaks
        
 
    '''
    # bins concat 
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    # x variables
    xs_all = bins['variable'].unique()
    # adjust all variables
    if not adj_all_var:
        bins2 = bins.loc[~((bins['bin'] == 'missing') & (bins['count_distr'] >= count_distr_limit))].reset_index(drop=True)
        bins2['badprob2'] = bins2.groupby('variable', group_keys=False).apply(lambda x: x['badprob'].shift(1)).reset_index(drop=True)
        bins2 = bins2.dropna(subset=['badprob2']).reset_index(drop=True)
        bins2 = bins2.assign(badprob_trend = lambda x: x.badprob >= x.badprob2)
        xs_adj = bins2.groupby('variable', group_keys=False)['badprob_trend'].nunique()
        xs_adj = xs_adj[xs_adj>1].index
    else:
        xs_adj = xs_all
    # length of adjusting variables
    xs_len = len(xs_adj)
    # special_values
    special_values = check_special_values(special_values, xs_adj)
    
    # breakslist of bins
    bins_breakslist = bins_to_breaks(bins,dt)
    # loop on adjusting variables
    if xs_len == 0:
        warnings.warn('The binning breaks of all variables are perfect according to default settings.')
        breaks_list = "{"+', '.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
        return breaks_list
    # else 
    def menu(i, xs_len, x_i):
        print('>>> Adjust breaks for ({}/{}) {}?'.format(i, xs_len, x_i))
        print('1: next \n2: yes \n3: back')
        adj_brk = input("Selection: ")
        while isinstance(adj_brk,str):
            if str(adj_brk).isdigit():
                adj_brk = int(adj_brk)
                if adj_brk not in [0,1,2,3]:
                    warnings.warn('Enter an item from the menu, or 0 to exit.')               
                    adj_brk = input("Selection: ")  
            else: 
                print('Input could not be converted to digit.')
                adj_brk = input("Selection: ") #update by ZK 
        return adj_brk
        
    # init param
    i = 1
    breaks_list = None
    while i <= xs_len:
        breaks = stop_limit = None
        # x_i
        x_i = xs_adj[i-1]
        sv_i = special_values[x_i] if (special_values is not None) and (x_i in special_values.keys()) else None
        # if sv_i is not None:
        #     sv_i = ','.join('\'')
        # basic information of x_i variable ------
        woebin_adj_print_basic_info(i, xs_adj, bins, dt, bins_breakslist)
        # adjusting breaks ------
        adj_brk = menu(i, xs_len, x_i)
        if adj_brk == 0: 
            return 
        while adj_brk == 2:
            # modify breaks adj_brk == 2
            breaks = input(">>> Enter modified breaks: ")
            breaks = re.sub("^[,\.]+|[,\.]+$", "", breaks)
            if breaks == 'N':
                stop_limit = 'N'
                breaks = None
            else:
                stop_limit = 0.1
            try:
                breaks = woebin_adj_break_plot(dt, y, x_i, breaks, stop_limit, sv_i, method=method)
            except:
                pass
            # adj breaks again
            adj_brk = menu(i, xs_len, x_i)
        if adj_brk == 3:
            # go back adj_brk == 3
            i = i-1 if i>1 else i
        else:
            # go next adj_brk == 1
            if breaks is not None and breaks != '': 
                bins_breakslist[x_i] = breaks
            i += 1
    # return 
    breaks_list = "{"+', '.join('\''+bins_breakslist.index[i]+'\': ['+bins_breakslist[i]+']' for i in np.arange(len(bins_breakslist)))+"}"
    if save_breaks_list is not None:
        bins_adj = woebin(dt, y, x=bins_breakslist.index, breaks_list=breaks_list)
        bins_to_breaks(bins_adj, dt, to_string=True, save_string=save_breaks_list)
    return breaks_list
    
