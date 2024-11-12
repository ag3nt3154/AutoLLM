import os

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
    
