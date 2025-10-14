# IPCV: Information-Preserving Compression for MLLM Visual Encoders

## Core Implementation 
The core logic of **IPCV** in Qwen2-VL is implemented in the class [`IPCV_ViT`](Qwen2-VL/Qwen2VL_IPCV/modeling_qwen2_vl_IPCV.py). 



## 🛠 Preparation

### Environment
```bash
conda create -n IPCV_Qwen2VL python=3.10 -y
conda activate IPCV_Qwen2VL
pip install accelerate qwen-vl-utils[decord]
pip install flash-attn --no-build-isolation
cd ../../lmms-eval && pip install -e .
pip install pytorch_memlab
pip install numpy==1.26.4 numexpr==2.12.1 pandas
```

> ⚠️ **Note**: If you face errors installing `flash-attn`,  
> please download a prebuilt wheel from [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).  
> Example:
> ```bash
> wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl
> pip install flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl
> ```


## 🐝 Examples of Evaluation
### Qwen2-VL
```bash
cd Qwen2-VL/transformers && pip install -e .
cd Qwen2-VL
bash eval_scripts/lmms_eval_multi.sh
```


### InternVL2
```bash
pip install -U transformers==4.55.4
cd InternVL2
bash eval_scripts/lmms_eval_multi.sh
```