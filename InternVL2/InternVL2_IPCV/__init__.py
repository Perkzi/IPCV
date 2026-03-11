# from .modeling_internvl_chat import InternVLChatModel

#from .modeling_internvl_chat_sparse_IPCV import InternVLChatModel
#from .modeling_internvl_chat_sparse_IPCV_FastV import InternVLChatModel

from .configuration_intern_vit import InternVisionConfig
from .configuration_internvl_chat import InternVLChatConfig

import importlib

def get_model_class(llm_method: str):
    if llm_method == "dart":
        module = importlib.import_module("InternVL2_IPCV.modeling_internvl_chat_sparse_IPCV")
    elif llm_method == "fastv":
        module = importlib.import_module("InternVL2_IPCV.modeling_internvl_chat_sparse_IPCV_FastV")
    else:
        raise ValueError(f"Unknown method: {llm_method}")

    return getattr(module, "InternVLChatModel")