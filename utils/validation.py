def validate_split(y_train, y_test):
    """
    Validates the train-test split for class distribution and sample size consistency.

    Parameters:
    ----------
    y_train : pd.Series
        Target values for the training set.
    
    y_test : pd.Series
        Target values for the test set.

    Functionality:
    -------------
    - Calculates total samples and expected 80/20 split sizes.
    - Prints the actual vs expected sizes of train/test sets.
    - Displays the class distribution in both sets.
    - Confirms the proportion of samples in each split.
    
    Useful for:
    -----------
    - Ensuring stratified sampling preserved class balance.
    - Double-checking correct split logic before modeling.
    """
    import numpy as np

    total_samples = len(y_train) + len(y_test)
    expected_train_size = int(0.8 * total_samples)
    expected_test_size = int(0.2 * total_samples)

    print("\n Validating Train-Test Split")
    print(" Total Samples:", total_samples)
    print(f" Expected Train Size (80%): {expected_train_size}, Actual: {len(y_train)}")
    print(f" Expected Test Size (20%): {expected_test_size}, Actual: {len(y_test)}")

    print("\n Train Distribution:\n", y_train.value_counts(normalize=True))
    print("\n Test Distribution:\n", y_test.value_counts(normalize=True))
    print("\n Ratio Check:")
    print(f"Train Set Proportion: {len(y_train) / total_samples:.2%}")
    print(f"Test Set Proportion: {len(y_test) / total_samples:.2%}")
