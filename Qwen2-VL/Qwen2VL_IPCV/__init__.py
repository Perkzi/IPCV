

# from .modeling_qwen2_vl_IPCV import Qwen2VLForConditionalGeneration

# from .modeling_qwen2_vl_IPCV_FastV import Qwen2VLForConditionalGeneration

from .configuration_qwen2_vl_ipcv import Qwen2VLConfig, Qwen2VLVisionConfig

import importlib

def get_model_class(llm_method: str):
    if llm_method == "dart":
        module = importlib.import_module("Qwen2VL_IPCV.modeling_qwen2_vl_IPCV")
    elif llm_method == "fastv":
        module = importlib.import_module("Qwen2VL_IPCV.modeling_qwen2_vl_IPCV_FastV")
    else:
        raise ValueError(f"Unknown method: {llm_method}")

    return getattr(module, "Qwen2VLForConditionalGeneration")
