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
        print("FlashAttention is not available.")
    return False
    
