import os
import subprocess
from unittest.mock import patch

import torch
from transformers.dynamic_module_utils import get_imports

if torch.backends.mps.is_available():
    # transformers.dynamic_module_utils.get_imports = fixed_get_imports
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    # transformers.dynamic_module_utils.get_imports = fixed_get_imports
    DEVICE = torch.device("cpu")


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
    else:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            func()

# def patched_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
#     from transformers.dynamic_module_utils import get_imports
#
#     with open(filename, "r", encoding="utf-8") as f:
#         content = f.read()
#
#     # filter out try/except block so in custom code we can have try/except imports
#     content = re.sub(r"\s*try\s*:\s*.*?\s*except\s*.*?:", "", content, flags=re.MULTILINE | re.DOTALL)
#
#     # Imports of the form `import xxx`
#     imports = re.findall(r"^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
#     # Imports of the form `from xxx import yyy`
#     imports += re.findall(r"^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
#     # Only keep the top-level module
#     imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]
#
#     final_imports = list(set(imports))
#
#     if not str(filename).endswith("modeling_florence2.py"):
#         return final_imports
#     else:
#         final_imports.remove("flash_attn")
#         return final_imports
