import torch
import torch.nn as nn

class KVFlopsMeter:
    def __init__(self, model):
        self.model = model
        self.total_flops = 0
        self.total_kv_MB = 0
        self.sample_count = 0
        self.hooks = []

    # def _attn_hook_llm(self, module, inputs, output):
    #     """
    #     LLM Attention hook: 计算 FLOPs + KV Cache
    #     """
    #     print("module type", module._get_name(),inputs,len(output))
    #     if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
    #         return
    #     x = inputs[0]

    def _attn_hook_llm(self, module, args, kwargs, output):
        #print("module type", module._get_name(),args, kwargs)
        if "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]

        #print("hidden_states shape:", x.shape)
    
        if x.dim() != 3:
            return
        B, Lq, D = x.shape
        num_heads = module.num_heads
        head_dim = module.head_dim

        # Q/K/V 投影 FLOPs
        proj_flops = 3 * (2 * B * Lq * D * (num_heads * head_dim))

        # Attention 矩阵乘法 FLOPs
        Lk = Lq
        past_kv = None
        
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            past_kv = output[2]
        else:
            past_kv = None


        #print("has get all",hasattr(past_kv, "get_all"),type(past_kv),)
        # 处理 tuple/list 形式
        if isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            k, v = past_kv

        # 处理 DynamicCache 形式
        # DynamicCache 形式
        elif past_kv is not None and hasattr(past_kv, "__len__") and hasattr(past_kv, "__getitem__"):
            if module.layer_idx is not None and len(past_kv) > module.layer_idx:
                #print("one layer kv",past_kv[module.layer_idx])
                k, v = past_kv[module.layer_idx]
        # elif hasattr(past_kv, "get_all"):  # DynamicCache API
        #     # get_all() 返回的是 [(k, v), (k, v), ...]，按 layer_idx 取
        #     all_kv = past_kv.get_all()
        #     print("allkv",all_kv,module.layer_idx,len(all_kv))
        #     if module.layer_idx is not None and module.layer_idx < len(all_kv):
        #         k, v = all_kv[module.layer_idx]
        #     else:
        #         k, v = None, None
        # 其它情况
        else:
            k, v = None, None

        
        #print("k,v",k.shape,v.shape,module.layer_idx)
        if isinstance(k, torch.Tensor) and k.dim() >= 3:
            Lk = k.shape[2]
            # 用 Lq > 1 判断 prefill
            if Lq > 1:
                kv_bytes = (k.numel() + v.numel()) * k.element_size()
                #print("single mb", kv_bytes/ (1024**2))
                self.total_kv_MB += kv_bytes / (1024**2)

        attn_flops = 2 * B * num_heads * Lq * Lk * head_dim * 2  # 两次乘法
        # 输出投影 FLOPs
        out_proj_flops = 2 * B * Lq * (num_heads * head_dim) * D

        total_flops = proj_flops + attn_flops + out_proj_flops
        self.total_flops += total_flops

    def _mlp_hook_llm(self, module, args, kwargs, output):
        """
        LLM MLP (SwiGLU) hook: 计算 FLOPs
        gate_proj: hidden_size -> intermediate_size
        up_proj: hidden_size -> intermediate_size
        down_proj: intermediate_size -> hidden_size
        """
        if "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]
        # if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
        #     return
        # x = inputs[0]
        #print("llm mlp hidden_states shape:", x.shape)
        if x.dim() != 3:
            return
        B, N, Din = x.shape  # Din = hidden_size

        # gate_proj 和 up_proj 是并行的两个 Linear
        Dout = module.intermediate_size
        flops_gate = 2 * B * N * Din * Dout
        flops_up   = 2 * B * N * Din * Dout

        # down_proj 是一个 Linear
        flops_down = 2 * B * N * Dout * Din

        total_flops = flops_gate + flops_up + flops_down
        self.total_flops += total_flops

    def _attn_hook_vit(self, module, args, kwargs, output):
        """
        ViT Attention hook: 只算 FLOPs，不算 KV Cache
        适配 VisionAttention / VisionFlashAttention2
        """
        if "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]
        # if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
        #     return
        # x = inputs[0]
        #print("attn vit hidden_states shape:", x.shape)
        # VisionAttention 输入是 [seq_len, dim]
        if x.dim() != 2:
            return
        seq_len, dim = x.shape
        B = 1  # ViT 这里是合并 batch 后的 seq_len
        num_heads = getattr(module, "num_heads", None)
        head_dim = getattr(module, "head_dim", None) or (dim // num_heads if num_heads else None)
        if num_heads is None or head_dim is None:
            return

        # Q/K/V 投影 FLOPs
        proj_flops = 3 * (2 * seq_len * dim * dim)

        # Attention 矩阵乘法 FLOPs（Lq = Lk = seq_len）
        attn_flops = 2 * B * num_heads * seq_len * seq_len * head_dim * 2  # 两次乘法

        # 输出投影 FLOPs
        out_proj_flops = 2 * seq_len * dim * dim

        total_flops = proj_flops + attn_flops + out_proj_flops
        self.total_flops += total_flops

    def _mlp_hook_vit(self, module, args, kwargs, output):
        """
        ViT MLP hook: 计算 FLOPs
        fc1: dim -> hidden_dim
        fc2: hidden_dim -> dim
        """
        if "hidden_states" in kwargs:
            x = kwargs["hidden_states"]
        else:
            if len(args) == 0 or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]
        #print("vit mlp hidden_states shape:", x.shape)
        # if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
        #     return
        # x = inputs[0]
        if x.dim() != 2 and x.dim() != 3:
            return

        if x.dim() == 3:
            B, N, Din = x.shape
            total_tokens = B * N
        else:
            total_tokens, Din = x.shape

        Dout = module.fc1.out_features

        # fc1 FLOPs
        flops_fc1 = 2 * total_tokens * Din * Dout
        # fc2 FLOPs
        flops_fc2 = 2 * total_tokens * Dout * Din

        total_flops = flops_fc1 + flops_fc2
        self.total_flops += total_flops


    def start(self):
        for name, m in self.model.named_modules():
            # 用类名字符串判断
            cls_name = m.__class__.__name__
            #print(name,cls_name)
            if cls_name in {"Qwen2VLAttention", "Qwen2VLFlashAttention2"}:
                self.hooks.append(m.register_forward_hook(self._attn_hook_llm, with_kwargs=True))
            elif cls_name == "Qwen2MLP":
                self.hooks.append(m.register_forward_hook(self._mlp_hook_llm, with_kwargs=True))
            elif cls_name in {"VisionAttention", "VisionFlashAttention2"}:
                self.hooks.append(m.register_forward_hook(self._attn_hook_vit, with_kwargs=True))
            elif cls_name == "VisionMlp":
                self.hooks.append(m.register_forward_hook(self._mlp_hook_vit, with_kwargs=True))



    def stop(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def record_sample(self):
        self.sample_count += 1
        self._counting_kv = True  # 每个样本开始时允许统计一次 KV

    def get_results(self):
        avg_flops = self.total_flops / self.sample_count if self.sample_count else 0
        avg_kv_MB = self.total_kv_MB / self.sample_count if self.sample_count else 0
        return avg_flops, avg_kv_MB