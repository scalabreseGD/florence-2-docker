import os
import re
from typing import Union, List

import torch
import transformers


def patched_get_imports(filename: Union[str, os.PathLike]) -> List[str]:
    from transformers.dynamic_module_utils import get_imports

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # filter out try/except block so in custom code we can have try/except imports
    content = re.sub(r"\s*try\s*:\s*.*?\s*except\s*.*?:", "", content, flags=re.MULTILINE | re.DOTALL)

    # Imports of the form `import xxx`
    imports = re.findall(r"^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]

    final_imports = list(set(imports))

    if not str(filename).endswith("modeling_florence2.py"):
        return final_imports
    else:
        final_imports.remove("flash_attn")
        return final_imports


if torch.backends.mps.is_available():
    transformers.dynamic_module_utils.get_imports = patched_get_imports
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    transformers.dynamic_module_utils.get_imports = patched_get_imports
    DEVICE = torch.device("cpu")
