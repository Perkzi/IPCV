
#!/bin/bash

# tasks=("gqa" "mmbench_en" "mmbench_cn" "mme" "pope" "seedbench" "textvqa" "vizwiz_vqa" "ocrbench")  
#        
# pruned_layers=(2 3 5)     
# reduction_ratios=(0.2 0.3 0.5) 

tasks=("mmbench_en")
pruned_layers=(3)
reduction_ratios=(0.65)
vit_pruned_layers=(7)
vit_reduction_ratios=(0.65)

for task in "${tasks[@]}"; do
  for pruned_layer in "${pruned_layers[@]}"; do
    for reduction_ratio in "${reduction_ratios[@]}"; do
      for vit_pruned_layer in "${vit_pruned_layers[@]}"; do
        for vit_reduction_ratio in "${vit_reduction_ratios[@]}"; do
          
          echo "========================================"
          echo "Current param group:"
          echo "Task: $task"
          echo "Pruned Layer: $pruned_layer"
          echo "Reduction Ratio: $reduction_ratio"
          echo "Vit Pruned Layer: $vit_pruned_layer"
          echo "Vit Reduction Ratio: $vit_reduction_ratio"
          echo "========================================"

          model_id="OpenGVLab/InternVL3-38B"
          model_name="InternVL3-38B"
          # model_id="OpenGVLab/InternVL3-8B"
          # model_name="InternVL3-8B"

          output_path="./logs/${model_name}/${task}/pruned_${pruned_layer}_ratio_${reduction_ratio}/vit_pruned_${vit_pruned_layer}_vit_ratio_${vit_reduction_ratio}/"
          
          mkdir -p "$output_path"

          # For IPCV: Sparse and vit_Sparse both should be True
          Sparse=True
          vit_Sparse=True
          AS_layer=3
          Top_K=10

          # For DART
          pivot_image_token=4
          pivot_text_token=4

          image_token_start_index=0
          image_token_length=0
          max_num_trunction=128

          torch_dtype=float16

          #GPU=0,1,2,3
          GPU=5

          # log file name
          log_file="${output_path}/run_detail.log"

          CUDA_VISIBLE_DEVICES=$GPU python3 -m accelerate.commands.launch \
              --num_processes=1 \
              --main_process_port 50008 \
              -m lmms_eval \
              --model internvl2_ipcv \
              --model_args pretrained=$model_id,device_map=auto,use_flash_attention_2=True,Sparse=$Sparse,vit_Sparse=$vit_Sparse,pruned_layer=$pruned_layer,vit_pruned_layer=$vit_pruned_layer,image_token_start_index=$image_token_start_index,image_token_length=$image_token_length,max_num_trunction=$max_num_trunction,reduction_ratio=$reduction_ratio,vit_reduction_ratio=$vit_reduction_ratio,pivot_image_token=$pivot_image_token,pivot_text_token=$pivot_text_token,torch_dtype=$torch_dtype,AS_layer=$AS_layer,Top_K=$Top_K \
              --tasks "${task}" \
              --batch_size 1 \
              --log_samples \
              --output_path "$output_path" 2>&1 | tee "$log_file"

          sleep 10
        done
      done
    done
  done
done
