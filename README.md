# Index of Functions and Classes

This document provides an overview of all the functions and classes included in the library, along with links to their detailed descriptions.

## General Index

### Functions

1. [corr_drop](#corr_drop)
2. [multicollinearity_check](#multicollinearity_check)
3. [find_columns_with_high_duplicates](#find_columns_with_high_duplicates)
4. [remove_high_vif_vars](#remove_high_vif_vars)

### Classes

1. [Transformation](#transformation)
2. [CapturePrint](#captureprint)

---

### Function Details

#### corr_drop
Removes highly correlated variables from a dataset based on a specified threshold.
- **File**: `mfa.py`
- **Args**:
  - `df`: Input DataFrame
  - `target`: Target variable to exclude
  - `threshold`: Correlation threshold
  - `how`: Method for dropping variables (`max_corr` or `order_list`)
  - `order_list`: Variable order for `order_list` method

[See Full Documentation](#corr_drop)

---

#### multicollinearity_check
Evaluates multicollinearity using Variance Inflation Factor (VIF).
- **File**: `mfa.py`
- **Args**:
  - `df`: Input DataFrame
  - `y`: Dependent variable/target column
  - `threshold`: VIF threshold for filtering
  - `only_final_vif`: Return only final VIF values

[See Full Documentation](#multicollinearity_check)

---

#### find_columns_with_high_duplicates
Identifies columns with a high percentage of duplicate values.
- **File**: `mfa.py`
- **Args**:
  - `df`: Input DataFrame
  - `threshold`: Minimum percentage of duplicate values to consider

[See Full Documentation](#find_columns_with_high_duplicates)

---

#### remove_high_vif_vars
Removes variables with high VIF to address multicollinearity.
- **File**: `mfa.py`
- **Args**:
  - `df`: Input DataFrame
  - `vif_threshold`: Threshold for VIF
  - `corr_threshold`: Correlation threshold
  - `dup_threshold`: Threshold for duplicate values
  - `target_col`: Target variable column
  - `how`: Method for handling correlations
  - `order_list`: Variable order preference

[See Full Documentation](#remove_high_vif_vars)

---

### Class Details

#### Transformation
Applies various data transformation techniques.
- **File**: `utils.py`

[See Full Documentation](#transformation)

---

#### CapturePrint
Captures printed output for logging or debugging.
- **File**: `utils.py`

[See Full Documentation](#captureprint)

---

For detailed usage and examples, refer to the [Full Documentation](functions_and_classes.md).


