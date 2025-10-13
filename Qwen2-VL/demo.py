from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import AutoTokenizer, AutoProcessor
from Qwen2VL_IPCV import Qwen2VLForConditionalGeneration,Qwen2VLConfig,Qwen2VLVisionConfig
from pytorch_memlab import profile

#@profile
def run_model():
    Sparse_config = {
                    "Sparse": True,
                    "pruned_layer": 3,
                    "reduction_ratio": 0.65,
                    "vit_Sparse": True,
                    "vit_pruned_layer": 3,
                    "vit_reduction_ratio": 0.65,

                    "image_token_start_index": 0,
                    "image_token_length": 0,
                    "max_num_trunction": 128,
                    "pivot_image_token": 4,
                    "pivot_text_token": 4,

                    "AS_layer": 7,
                }
    
    # Load the model in half-precision on the available device(s)
    vision_config = Qwen2VLVisionConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",Sparse_config = Sparse_config)

    config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", 
        Sparse_config=Sparse_config,
        vision_config = vision_config)
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            config = config,torch_dtype="auto", 
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Image
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    print("CUDA memory used before processing images to standard form: ", torch.cuda.memory_allocated() / 1024**2, "MB")
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    ) 
    # inputs['pixel_values']: [grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size]

    inputs = inputs.to("cuda")
    print("CUDA memory used after processing images to standard form: ", torch.cuda.memory_allocated() / 1024**2, "MB")

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128) # inputs: (['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text)

run_model()