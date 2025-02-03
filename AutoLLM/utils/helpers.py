import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_attr(config, key, default_val):
    """
    Get the value of a key from a dictionary, or return a default value if the key is not found.

    Args:
        config (dict): The dictionary to search for the key.
        key (str): The key to search for.
        default_val: The default value to return if the key is not found.
    """
    if config is None:
        return default_val
    return config.get(key, default_val)


def is_flash_attention_available():
    try:
        import flash_attn
        version = flash_attn.__version__
        print(f'Flash Attention version: {version}')
        if not version.startswith("2"):
            print(f"FlashAttention version {version} is available, but it is not version 2.")
            return 1
        else:
            return 2
    except ImportError:
        print("Flash Attention is not available.")
    return False


def get_file_ext(file_path, check_ext=None):
    """
    Check if the file has a specific extension.
    
    Args:
        file_path (str): Path to the file.
        extension (str): Expected file extension, e.g., '.txt', '.jpg'.

    Returns:
        bool: True if the file has the specified extension, otherwise False.
    """
    # Extract the file extension
    _, ext = os.path.splitext(file_path)

    if check_ext is not None:
        return ext.lower() == check_ext.lower()
    
    # Check if it matches the provided extension
    return ext.lower()
    


def load_df(df_path):
    '''
    Load dataframe from csv or excel files
    '''

    file_type = df_path.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(df_path)
    elif file_type == 'xlsx':
        df = pd.read_excel(df_path)
    else:
        assert False, 'WRONG FILE TYPE!'

    try:
        df.drop('Unnamed: 0', axis='columns', inplace=True)
    except KeyError:
        pass
    return df



def create_folder(folder_path):
    '''
    Creates folder at folder_path.
    Does not create anything if folder exists
    '''
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    return





def split_dataframe(df, test_size=0.2, random_state=None, stratify_col=None):
    """
    Splits a DataFrame into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
        random_state (int): Seed used by the random number generator (default is None).
        stratify_col (str): Column to stratify by for stratified sampling (default is None).

    Returns:
        pd.DataFrame, pd.DataFrame: Training and testing DataFrames.
    """
    stratify = df[stratify_col] if stratify_col else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return train_df, test_df