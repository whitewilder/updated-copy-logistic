# -*- coding: utf-8 -*-

import pandas as pd
from pandas.api.types import CategoricalDtype
import pkg_resources


def germancredit():
    '''
    German Credit Data
    ------
    Credit data that classifies debtors described by a set of 
    attributes as good or bad credit risks. See source link 
    below for detailed information.
    [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
    
    Params
    ------
    
    Returns
    ------
    DataFrame
    
    Examples
    ------
    import scorecardpy as sc
    
    # load data
    dat = sc.germancredit()
    
    # # data structure
    # dat.shape
    # dat.dtypes
    '''
    DATA_FILE = pkg_resources.resource_filename('scorecardpy', 'data/germancredit.csv')
    
    dat = pd.read_csv(DATA_FILE)
    # categorical levels
    cate_levels = {
            "status_of_existing_checking_account": ['... < 0 DM', '0 <= ... < 200 DM', '... >= 200 DM / salary assignments for at least 1 year', 'no checking account'], 
            "credit_history": ["no credits taken/ all credits paid back duly", "all credits at this bank paid back duly", "existing credits paid back duly till now", "delay in paying off in the past", "critical account/ other credits existing (not at this bank)"], 
            "savings_account_and_bonds": ["... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM", "unknown/ no savings account"],
            "present_employment_since": ["unemployed", "... < 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", "... >= 7 years"], 
            "personal_status_and_sex": ["male : divorced/separated", "female : divorced/separated/married", "male : single", "male : married/widowed", "female : single"], 
            "other_debtors_or_guarantors": ["none", "co-applicant", "guarantor"], 
            "property": ["real estate",  "building society savings agreement/ life insurance",  "car or other, not in attribute Savings account/bonds",  "unknown / no property"],
            "other_installment_plans": ["bank", "stores", "none"],
            "housing": ["rent", "own", "for free"], 
            "job": ["unemployed/ unskilled - non-resident", "unskilled - resident", "skilled employee / official", "management/ self-employed/ highly qualified employee/ officer"],
            "telephone": ["none", "yes, registered under the customers name"], 
            "foreign_worker": ["yes", "no"]}
    # func of cate
    def cate_type(levels):
        return CategoricalDtype(categories=levels, ordered=True)
    # to cate
    for i in cate_levels.keys():
        dat[i] = dat[i].astype(cate_type(cate_levels[i]))
    # return
    return dat


'''
# datasets
import scorecardpy as sc
dat1 = sc.germancredit()
dat1 = check_y(dat1, 'creditability', 'bad|1')
dat2 = pd.DataFrame({'creditability':[0,1]}).sample(50, replace=True)
# dat2 = pd.DataFrame({'creditability':np.random.choice([0,1], 50)})
dat = pd.concat([dat2, dat1], ignore_index=True)


y = "creditability"
x_i = "duration.in.month"
dtm = pd.DataFrame({'y':dat[y], 'variable':x_i, 'value':dt[x_i]})

###### dtm ######
# y
y = dat['creditability']

# x
# numerical data
xvar =  "credit_amount" # "foreign_worker # 'age_in_years' #'number_of_existing_credits_at_this_bank' # 
x= dat1[xvar]
spl_val = [2600, 9960, "6850%,%missing"]
breaks = [2000, 4000, 6000]
breaks = ['26%,%missing', 28, 35, 37]

# categorical data
xvar= 'purpose'#'housing' # "job" # "credit_amount"; #
x= dat[xvar] # pd.Categorical(dat[xvar], categories=['rent', 'own','for free']) 
breaks = ["own", "for free%,%rent%,%missing"]
breaks = ["own", "for free%,%rent"]


dtm = pd.DataFrame({'y':y, 'variable':xvar, 'value':x})
# dtm.value = None
'''



def calculate_vif(df):
    """Calculate VIF for the dataframe."""
    # Add constant to the dataframe for intercept
    df_with_const = add_constant(df)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])]
    return vif_data

def remove_high_vif_vars(df, vif_threshold):
    """
    Remove variables with VIF above the threshold iteratively.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with all variables.
    vif_threshold : float
        The VIF threshold above which variables will be considered for removal.

    Returns
    -------
    pandas.DataFrame
        DataFrame with variables having VIF above the threshold removed.
    """
    # Copy the dataframe to avoid modifying the original
    df_clean = df.copy()
    variables = df_clean.columns.tolist()

    while True:
        # Calculate VIF for current variables
        df_vif = calculate_vif(df_clean[variables])
        
        # Identify variables with high VIF
        high_vif_vars = df_vif[df_vif["VIF"] > vif_threshold]
        
        if high_vif_vars.empty:
            # Exit loop if no variables exceed the VIF threshold
            break
        
        # Remove the variable with the highest VIF
        var_to_remove = high_vif_vars.sort_values(by="VIF", ascending=False).iloc[0]["Variable"]
        
        if var_to_remove in variables:
            variables.remove(var_to_remove)
            df_clean = df_clean[variables]
        else:
            break

    # Print removed variables
    removed_vars = set(df.columns) - set(variables)
    if removed_vars:
        print(f"Removed variables: {removed_vars}")

    return df_clean
# Example DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, 7, 8, 7],
    'C': [2, 4, 6, 8, 10],
    'D': [10, 12, 14, 16, 18],
    'E': [1, 2, 3, 49, 5],
}
df = pd.DataFrame(data)
# Remove multicollinearity
df_cleaned = calculate_vif(df)
print(df_cleaned)


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def remove_multicollinearity(df, vif_threshold=10, corr_threshold=0.8):
    """
    Remove variables from the dataframe to reduce multicollinearity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with all variables.
    vif_threshold : float
        The VIF threshold above which variables will be considered for removal.
    corr_threshold : float
        The correlation coefficient threshold above which variables will be considered highly correlated.

    Returns
    -------
    pandas.DataFrame
        DataFrame with multicollinear variables removed.
    """
    # Function to calculate VIF
    def calculate_vif(df):
        # Add constant to the dataframe for intercept
        df = add_constant(df)
        vif_data = pd.DataFrame()
        vif_data["Variable"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif_data

    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Find pairs of highly correlated variables
    high_corr_vars = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                high_corr_vars.add(corr_matrix.columns[i])
                high_corr_vars.add(corr_matrix.columns[j])

    # Start with all variables
    variables = df.columns.tolist()
    removed_vars = set()

    while True:
        # Calculate VIF for current variables
        df_vif = calculate_vif(df[variables])
        high_vif_vars = df_vif[df_vif["VIF"] > vif_threshold]["Variable"].tolist()
        
        # Combine high correlation and high VIF variables
        vars_to_remove = set(high_corr_vars).intersection(variables)
        vars_to_remove.update(high_vif_vars)
        
        if not vars_to_remove:
            break

        # Remove variables with the highest VIF
        if vars_to_remove:
            var_to_remove = df_vif.loc[df_vif["Variable"].isin(vars_to_remove), "VIF"].idxmax()
            if var_to_remove in variables:
                variables.remove(var_to_remove)
                removed_vars.add(var_to_remove)
                high_corr_vars.discard(var_to_remove)
        else:
            break

    # Print removed variables
    if removed_vars:
        print(f"Removed variables: {removed_vars}")

    return df[variables]


# Example DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 6, 7, 8, 7],
    'C': [2, 4, 6, 8, 10],
    'D': [10, 12, 14, 16, 18]
}
df = pd.DataFrame(data)
# Remove multicollinearity
df_cleaned = remove_multicollinearity(df, vif_threshold=10, corr_threshold=0.8)
print(df_cleaned)

