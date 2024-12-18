# Function Documentation for Data Preprocessing and Multicollinearity Check

## Table of Contents

1. [corr\_drop](#corr_drop)
2. [multicollinearity\_check](#multicollinearity_check)
3. [find\_columns\_with\_high\_duplicates](#find_columns_with_high_duplicates)
4. [remove\_high\_vif\_vars](#remove_high_vif_vars)

---

## 1. corr\_drop

### Description

Removes highly correlated variables from a dataset while preserving the target variable.

### Function Signature

```python
def corr_drop(df, target, threshold=0.9, how='max_corr', order_list=None):
```

### Parameters

- `df (pd.DataFrame)`: The input dataset.
- `target (str)`: The target variable's column name.
- `threshold (float)`: Correlation threshold to identify highly correlated variables. Default is 0.9.
- `how (str)`: Method for removing correlated variables:
  - `'max_corr'`: Removes the variable with the highest correlation count.
  - `'order_list'`: Uses the specified order of importance in `order_list` to remove variables.
- `order_list (list, optional)`: Specifies the importance order for variable removal. Used when `how='order_list'`.

### Returns

- `pd.DataFrame`: A DataFrame with highly correlated variables removed.

### Example

```python
import pandas as pd

data = {
    'A': [1, 2, 3, 4],
    'B': [2, 4, 6, 8],
    'C': [1, 1, 2, 2],
    'target': [0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Remove highly correlated variables
result = corr_drop(df, target='target', threshold=0.8, how='max_corr')
print(result)
```

---

## 2. multicollinearity\_check

### Description

Checks for multicollinearity among variables using the Generalized Variance Inflation Factor (GVIF).

### Function Signature

```python
def multicollinearity_check(df, y, threshold=18, only_final_vif=True):
```

### Parameters

- `df (pd.DataFrame)`: The input dataset.
- `y (str)`: Dependent variable or target column name.
- `threshold (float)`: Threshold for VIF values to identify multicollinearity. Default is 18.
- `only_final_vif (bool)`: If `True`, returns only the final VIF values. Default is `True`.

### Returns

- `tuple`: A tuple containing:
  1. `pd.DataFrame`: GVIF values.
  2. `dict`: Columns exceeding the threshold.
  3. `pd.DataFrame`: Filtered DataFrame with multicollinear variables removed.

### Example

```python
# Example DataFrame
data = {
    'X1': [1, 2, 3, 4],
    'X2': [2, 4, 6, 8],
    'X3': [5, 5, 6, 6],
    'Y': [0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Check multicollinearity
vif_values, filtered_columns, filtered_df = multicollinearity_check(df, y='Y', threshold=10)
print(vif_values)
```

---

## 3. find\_columns\_with\_high\_duplicates

### Description

Identifies columns with a high percentage of duplicate values in a dataset.

### Function Signature

```python
def find_columns_with_high_duplicates(df, threshold=0.5):
```

### Parameters

- `df (pd.DataFrame)`: The input dataset.
- `threshold (float)`: Minimum percentage of duplicate values for a column to be flagged. Default is 0.5.

### Returns

- `list`: A list of column names with a high percentage of duplicates.

### Example

```python
# Example DataFrame
data = {
    'A': [1, 1, 1, 1],
    'B': [1, 2, 3, 4],
    'C': [5, 5, 6, 6]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Find columns with high duplicate ratios
high_dup_columns = find_columns_with_high_duplicates(df, threshold=0.6)
print(high_dup_columns)
```

---

## 4. remove\_high\_vif\_vars

### Description

Removes variables with high Variance Inflation Factor (VIF) and other redundant variables to address multicollinearity.

### Function Signature

```python
def remove_high_vif_vars(df, vif_threshold=7, corr_threshold=0.9, dup_threshold=0.7, target_col='intodef', how='order_list', order_list=None):
```

### Parameters

- `df (pd.DataFrame)`: The input dataset containing independent and dependent variables.
- `vif_threshold (float)`: Threshold for VIF values. Default is 7.
- `corr_threshold (float)`: Correlation threshold for redundancy. Default is 0.9.
- `dup_threshold (float)`: Threshold for duplicate values. Default is 0.7.
- `target_col (str)`: Name of the dependent variable. Default is `'intodef'`.
- `how (str)`: Strategy for handling correlations:
  - `'order_list'`: Uses `order_list` to remove variables.
- `order_list (list, optional)`: Variable priority list for removal.

### Returns

- `tuple`:
  1. `pd.DataFrame`: Cleaned DataFrame.
  2. `pd.DataFrame`: Final VIF values.

### Example

```python
# Example DataFrame
data = {
    'X1': [1, 2, 3, 4],
    'X2': [2, 4, 6, 8],
    'X3': [1, 1, 2, 2],
    'target': [0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Remove high VIF variables
cleaned_df, final_vif = remove_high_vif_vars(df, vif_threshold=5, target_col='target')
print(cleaned_df)
print(final_vif)
```

