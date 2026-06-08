import gc

import torch
from transformers import AutoTokenizer


def get_device():
    """
    Dynamically determines the best available compute device.
    
    Returns:
    str: "cuda" for NVIDIA GPUs (Vast.ai), "mps" for Apple Silicon Macs, or "cpu" as fallback.
    """
    if torch.cuda.is_available():
        # Executes automatically on your Vast.ai NVIDIA instances
        return "cuda"
    elif torch.backends.mps.is_available():
        # Executes automatically on your local Mac Apple Silicon GPU
        return "mps"
    else:
        return "cpu"


def supports_bf16(device: str = None):
    """
    Checks if the specified device supports bfloat16 precision.
    
    Parameters:
    device (str): The device to check ("cuda", "mps", "cpu", or None for auto-detection).
    
    Returns:
    bool: True if bf16 is supported, False otherwise.
    """
    if device is None:
        device = get_device()
    
    if device == "cuda":
        return torch.cuda.is_bf16_supported()
    elif device == "mps":
        # MPS generally supports bf16, but let's be cautious
        try:
            return torch.backends.mps.is_available()
        except:
            return False
    else:
        # CPU doesn't typically support bf16 in PyTorch
        return False


def get_tokenizer(model_name: str):
    """
    This function returns a tokenizer based on the model name.

    Parameters:
    model_name: The name of the model for which the tokenizer is needed.

    Returns:
    A tokenizer for the specified model.
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def clear_mem(verbose: bool = False):
    """
    This function is used to clear the memory allocated by PyTorch.
    It calls the garbage collector and clears GPU memory if available.
    After clearing the memory, it prints the current amount of memory still allocated.

    Parameters:
    verbose (bool): Whether to print additional information.
    """

    gc.collect()
    device = get_device()
    
    # Clear device-specific memory
    if device == "cuda":
        torch.cuda.empty_cache()
        print(
            f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB"
        )
    elif device == "mps":
        # MPS doesn't have explicit memory clearing, but we can log available memory
        print("Running on Apple Silicon MPS")
    else:
        print("Running on CPU")

    if verbose:

        def try_attr(x, a):
            try:
                return getattr(x, a)
            except:
                # amazing that this can cause...
                # (AttributeError, OSError, AssertionError, RuntimeError, ModuleNotFoundError)
                return None

        for obj in gc.get_objects():
            if torch.is_tensor(obj) or torch.is_tensor(try_attr(obj, "data")):
                print(type(obj), obj.size(), obj.dtype)
