import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def delete_duplicates(df, method='all_columns', keep='first', columns=None):
    """
    Delete duplicate rows from a DataFrame based on specified parameters.
    
    Args:
        df: pandas DataFrame
        method: 'all_columns' (compare all columns) or 'selected_columns' (compare only specified columns)
        keep: 'first' (keep first occurrence), 'last' (keep last occurrence), or 'none' (remove all duplicates)
        columns: list of columns to consider (only used when method is 'selected_columns')
    
    Returns:
        Tuple of (processed DataFrame, message)
    """
    try:
        original_rows = len(df)
        
        if method == 'all_columns':
            subset = None  # Compare all columns
        elif method == 'selected_columns':
            if not columns:
                return df, "No columns selected for duplicate check"
            subset = columns
        else:
            return df, f"Invalid method '{method}'. Use 'all_columns' or 'selected_columns'"
        
        if keep == 'none':
            # Keep only rows that are not duplicates at all
            df = df.drop_duplicates(subset=subset, keep=False)
        else:
            # Keep first or last occurrence of duplicates
            df = df.drop_duplicates(subset=subset, keep=keep)
            
        removed_rows = original_rows - len(df)
        message = f"Removed {removed_rows} duplicate rows (method: {method}, keep: {keep})"
        
        return df, message
        
    except Exception as e:
        return df, f"Error deleting duplicates: {str(e)}"


def encode_categorical_column(df, column, method="onehot"):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is numeric and cannot be encoded as categorical.")

    if method == "onehot":
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=False).astype(int)
        df = df.drop(columns=[column])
        df = pd.concat([df, dummies], axis=1)

    elif method == "label":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

    elif method == "ordinal":
        oe = OrdinalEncoder()
        df[column] = oe.fit_transform(df[[column]])

    else:
        raise ValueError("Unsupported encoding method. Use 'onehot', 'label', or 'ordinal'.")

    return df

def handle_missing_values(df, column, method):
    """Handle missing values in a DataFrame column using specified method."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
        
    if method == "drop":
        return df.dropna(subset=[column])
    
    if method in ["mean", "median"]:
        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].mean() if method == "mean" else df[column].median()
            df[column] = df[column].fillna(fill_value)
    elif method == "mode":
        mode_val = df[column].mode()[0] if not df[column].mode().empty else None
        if mode_val is not None:
            df[column] = df[column].fillna(mode_val)
    elif method == "forward_fill":
        df[column] = df[column].ffill()
    elif method == "backward_fill":
        df[column] = df[column].bfill()
    
    return df

def remove_outliers(df, column, method="IQR"):
    """
    Remove outliers from a DataFrame column using specified method.
    
    Args:
        df: pandas DataFrame
        column: column name to process
        method: one of ["IQR", "Z-score", "percentile"]
    
    Returns:
        DataFrame with outliers removed
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df
    
    if method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == "Z-score":
        if df[column].std() == 0:  # Avoid division by zero
            return df
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores <= 3]
    
    elif method == "percentile":
        lower = df[column].quantile(0.05)  # More common to use 5% and 95%
        upper = df[column].quantile(0.95)
        return df[(df[column] >= lower) & (df[column] <= upper)]
    
    return df

def fix_data_types(df, column, dtype):
    """
    Convert column to specified data type with proper error handling.
    
    Args:
        df: pandas DataFrame
        column: column name to convert
        dtype: one of ["int", "float", "str", "category", "bool"]
    
    Returns:
        DataFrame with converted column
    """
    try:
        if dtype == "int":
            # First ensure the column is numeric, then round before converting to int
            # This handles decimal values properly by rounding them
            df[column] = pd.to_numeric(df[column], errors='coerce')
            # Round to nearest integer before converting to handle decimals like 3.6 -> 4
            df[column] = df[column].round().astype('Int64')
        elif dtype == "float":
            df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)
        elif dtype == "str":
            df[column] = df[column].astype(str)
        elif dtype == "category":
            df[column] = df[column].astype('category')
        elif dtype == "bool":
            # Handle boolean conversion more carefully
            if df[column].dtype == 'object':
                # Convert string representations to boolean
                df[column] = df[column].map({
                    'True': True, 'False': False, 'true': True, 'false': False, 
                    '1': True, '0': False, 1: True, 0: False
                })
            else:
                df[column] = df[column].astype(bool)
    except Exception as e:
        print(f"Error converting column {column} to {dtype}: {str(e)}")
        raise e  # Re-raise the exception so it can be handled by the calling function
    
    return df

def group_by_columns(df, group_by_column, aggregate_column, aggregation_method="list"):
    """
    Group DataFrame by one column and aggregate another column.
    
    Args:
        df: pandas DataFrame
        group_by_column: column to group by (e.g., 'BillNo')
        aggregate_column: column to aggregate (e.g., 'Items')
        aggregation_method: one of ["list", "count", "sum", "mean", "first", "last", "unique_list"]
    
    Returns:
        DataFrame with grouped results
    """
    if group_by_column not in df.columns:
        raise ValueError(f"Group by column '{group_by_column}' not found in DataFrame")
    
    if aggregate_column not in df.columns:
        raise ValueError(f"Aggregate column '{aggregate_column}' not found in DataFrame")
    
    try:
        if aggregation_method == "list":
            # Create a list of all values for each group
            result = df.groupby(group_by_column)[aggregate_column].apply(
                lambda x: ', '.join(x.astype(str).tolist())
            ).reset_index()
            result.columns = [group_by_column, f"{aggregate_column}_list"]
            
        elif aggregation_method == "unique_list":
            # Create a list of unique values for each group
            result = df.groupby(group_by_column)[aggregate_column].apply(
                lambda x: ', '.join(x.astype(str).unique().tolist())
            ).reset_index()
            result.columns = [group_by_column, f"{aggregate_column}_unique_list"]
            
        elif aggregation_method == "count":
            # Count occurrences
            result = df.groupby(group_by_column)[aggregate_column].count().reset_index()
            result.columns = [group_by_column, f"{aggregate_column}_count"]
            
        elif aggregation_method == "sum":
            # Sum numeric values
            if pd.api.types.is_numeric_dtype(df[aggregate_column]):
                result = df.groupby(group_by_column)[aggregate_column].sum().reset_index()
                result.columns = [group_by_column, f"{aggregate_column}_sum"]
            else:
                raise ValueError(f"Cannot sum non-numeric column '{aggregate_column}'")
                
        elif aggregation_method == "mean":
            # Mean of numeric values
            if pd.api.types.is_numeric_dtype(df[aggregate_column]):
                result = df.groupby(group_by_column)[aggregate_column].mean().reset_index()
                result.columns = [group_by_column, f"{aggregate_column}_mean"]
            else:
                raise ValueError(f"Cannot calculate mean of non-numeric column '{aggregate_column}'")
                
        elif aggregation_method == "first":
            # First value in each group
            result = df.groupby(group_by_column)[aggregate_column].first().reset_index()
            result.columns = [group_by_column, f"{aggregate_column}_first"]
            
        elif aggregation_method == "last":
            # Last value in each group
            result = df.groupby(group_by_column)[aggregate_column].last().reset_index()
            result.columns = [group_by_column, f"{aggregate_column}_last"]
            
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        # Add other columns from the original dataframe (using first occurrence)
        other_columns = [col for col in df.columns if col not in [group_by_column, aggregate_column]]
        if other_columns:
            other_data = df.groupby(group_by_column)[other_columns].first().reset_index()
            result = pd.merge(result, other_data, on=group_by_column, how='left')
        
        return result
        
    except Exception as e:
        print(f"Error in group_by_columns: {str(e)}")
        raise e