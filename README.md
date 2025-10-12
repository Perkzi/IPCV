## IPCV 核心实现
ViT-Prunning-self/Qwen2-VL/Qwen2VL_DART_ViT/modeling_qwen2_vl_dart_vit_mean_shift_similarity_kv.py



## 🛠 Preparation
```bash
git clone -b vit_mean_shift git@github.com:shuangchen2003/ViT-Prunning-self.git
```

### Qwen2-VL
```bash
 conda create -n DART_Qwen2VL python=3.10 -y
 conda activate DART_Qwen2VL
 cd Qwen2-VL/transformers && pip install -e .
 pip install accelerate qwen-vl-utils[decord]
 pip install flash-attn --no-build-isolation
 cd ../../lmms-eval && pip install -e .
 pip install pytorch_memlab
```


### Qwen2-VL
### 🐝 Examples
```bash
cd Qwen2-VL
bash eval_scripts/lmms_eval_multi.sh
```



