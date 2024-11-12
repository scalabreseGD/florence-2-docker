import os
import subprocess
from unittest.mock import patch

import torch
from transformers.dynamic_module_utils import get_imports


def __init_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    return device


DEVICE = __init_device()


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def __install_flash_attn():
    subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"},
                   shell=True)


def run_with_patch(func, if_cuda_func=__install_flash_attn):
    if DEVICE == torch.device("cuda"):
        if_cuda_func()
        func()
    else:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            func()
