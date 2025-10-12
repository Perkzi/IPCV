import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset, Dataset
from tqdm import tqdm
import re
import json
import os
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
from Qwen2VL_DART_ViT import Qwen2VLForConditionalGeneration
import time

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_output_directory(base_dir="results"):
    """创建带时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"textvqa_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def normalize_answer(text):
    """答案标准化：小写转换、去除标点、多余空格和冠词"""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)  # 移除冠词
    text = re.sub(r"[^\w\s]", "", text)          # 移除非字母数字字符
    text = re.sub(r"\s+", " ", text)             # 合并多余空格
    return text.strip()

def calculate_metrics(predictions, references):
    """计算VQA多种指标"""
    results = {
        'accuracy': 0,
        'exact_match': 0,
        'per_question_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_answer_length': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    for pred, refs in zip(predictions, references):
        pred_norm = normalize_answer(pred)
        refs_norm = [normalize_answer(ref) for ref in refs]
        
        # 基本准确率
        if pred_norm in refs_norm:
            results['accuracy'] += 1
        
        # 完全匹配
        if any(pred_norm == ref_norm for ref_norm in refs_norm):
            results['exact_match'] += 1
        
        # 按问题类型统计
        q_type = "other"
        if "what" in refs[0].lower():
            q_type = "what"
        elif "how" in refs[0].lower():
            q_type = "how"
        elif "why" in refs[0].lower():
            q_type = "why"
        elif "where" in refs[0].lower():
            q_type = "where"
        
        results['per_question_type'][q_type]['total'] += 1
        if pred_norm in refs_norm:
            results['per_question_type'][q_type]['correct'] += 1
        
        # 按答案长度统计
        avg_ref_len = np.mean([len(ref) for ref in refs])
        length_bucket = "short" if avg_ref_len < 4 else "medium" if avg_ref_len < 8 else "long"
        results['per_answer_length'][length_bucket]['total'] += 1
        if pred_norm in refs_norm:
            results['per_answer_length'][length_bucket]['correct'] += 1
    
    # 转换为百分比
    total_samples = len(predictions)
    results['accuracy'] = results['accuracy'] / total_samples * 100
    results['exact_match'] = results['exact_match'] / total_samples * 100
    
    # 计算子类准确率
    for q_type in results['per_question_type']:
        correct = results['per_question_type'][q_type]['correct']
        total = results['per_question_type'][q_type]['total']
        results['per_question_type'][q_type]['accuracy'] = correct / total * 100 if total > 0 else 0
    
    for length_bucket in results['per_answer_length']:
        correct = results['per_answer_length'][length_bucket]['correct']
        total = results['per_answer_length'][length_bucket]['total']
        results['per_answer_length'][length_bucket]['accuracy'] = correct / total * 100 if total > 0 else 0
    
    return results

def evaluate_textvqa(model, processor, dataset, output_dir, num_samples=1000, save_samples=100):
    """
    执行TextVQA评估
    :param model: 加载的模型
    :param processor: 数据处理器
    :param dataset: 数据集对象
    :param output_dir: 输出目录
    :param num_samples: 评估样本数
    :param save_samples: 保存详细结果的样本数
    """
    model.eval()
    predictions = []
    references = []
    sample_results = []
    question_ids = []
    
    # 限制评估样本数量
    eval_dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    total_time = 0
    for idx, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        try:
            # 预处理数据
            image = example["image"].convert("RGB")
            question = example["question"]
            question_id = example.get("question_id", f"q_{idx}")
            
            # 模型输入处理
            inputs = processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(device)
            
            start = time.time()
            # 生成答案
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    num_beams=5,
                    early_stopping=True
                )
            end = time.time()
            total_time += end - start
            
            # 解码答案
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 后处理：移除问题文本和特殊token
            answer = answer.replace(question, "").strip()
            answer = re.sub(r"<\|endoftext\|>.*", "", answer)  # 移除结束符后内容
            
            predictions.append(answer)
            references.append(example["answers"])
            question_ids.append(question_id)
            
            # 保存部分样本的详细结果
            if idx < save_samples:
                sample_results.append({
                    "question_id": question_id,
                    "question": question,
                    "prediction": answer,
                    "references": example["answers"],
                    "image_path": example.get("image_path", f"image_{idx}.jpg")
                })
                
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            predictions.append("")
            references.append(example["answers"])
            question_ids.append(f"q_{idx}")
    print("总运行时间为：",total_time,"s, ","平均运行时间为:",total_time/(idx+1),"s")
    # 计算指标
    metrics = calculate_metrics(predictions, references)
    
    # 保存结果
    save_results(output_dir, sample_results, metrics, predictions, references, question_ids)
    
    return metrics

def save_results(output_dir, sample_results, metrics, all_predictions, all_references, question_ids):
    """保存所有结果到文件"""
    # 保存详细样本结果
    with open(os.path.join(output_dir, "sample_results.json"), "w") as f:
        json.dump(sample_results, f, indent=2)
    
    # 保存所有预测结果
    all_results = [{
        "question_id": qid,
        "prediction": pred,
        "references": refs
    } for qid, pred, refs in zip(question_ids, all_predictions, all_references)]
    
    with open(os.path.join(output_dir, "all_predictions.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 保存评估指标
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # 保存人类可读的摘要
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"TextVQA Evaluation Report\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(all_predictions)}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Exact Match: {metrics['exact_match']:.2f}%\n\n")
        
        f.write("Per Question Type Accuracy:\n")
        for q_type, stats in metrics['per_question_type'].items():
            f.write(f"- {q_type.capitalize()}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})\n")
        
        f.write("\nPer Answer Length Accuracy:\n")
        for length, stats in metrics['per_answer_length'].items():
            f.write(f"- {length.capitalize()}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})\n")
        
        f.write("\nSample Predictions:\n")
        for i, sample in enumerate(sample_results[:5]):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"Question: {sample['question']}\n")
            f.write(f"Prediction: {sample['prediction']}\n")
            f.write(f"References: {', '.join(sample['references'])}\n")
    
    print(f"结果已保存到: {output_dir}")

def load_or_download_dataset(dataset_name="textvqa", split="validation", cache_dir="data"):
    """加载或下载数据集"""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        print(f"数据集已加载，来自缓存: {cache_dir}")
    except:
        print("下载数据集...")
        dataset = load_dataset(dataset_name, split=split)
        dataset.save_to_disk(os.path.join(cache_dir, dataset_name))
        print(f"数据集已保存到: {os.path.join(cache_dir, dataset_name)}")
    return dataset

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen2VL TextVQA 评估脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL", 
                        help="模型名称或路径")
    parser.add_argument("--dataset", type=str, default="textvqa", 
                        help="数据集名称或路径")
    parser.add_argument("--split", type=str, default="validation", 
                        help="数据集划分 (e.g., validation, test)")
    parser.add_argument("--num_samples", type=int, default=500, 
                        help="评估样本数量")
    parser.add_argument("--save_samples", type=int, default=100, 
                        help="保存详细结果的样本数量")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="结果输出目录")
    parser.add_argument("--cache_dir", type=str, default="data", 
                        help="数据集缓存目录")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    DART_config = {'pruned_layer':2,
                    'image_token_start_index':0,
                    'image_token_length':0,
                    'max_num_trunction':128,
                    'reduction_ratio':0.2,
                    'pivot_image_token':4,
                    'pivot_text_token':4,
                    'K':2,
                    'random_choose':False}
    # 创建输出目录
    output_dir = create_output_directory(args.output_dir)
    
    # 保存运行配置
    config = vars(args)
    config['eval_date'] = datetime.now().isoformat()
    config['device'] = device
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("加载模型和处理器...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).to(device)
    model.visual.config.DART_config = DART_config
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_or_download_dataset(
        args.dataset, 
        args.split, 
        args.cache_dir
    )
    
    # 执行评估
    print("开始评估...")
    results = evaluate_textvqa(
        model, 
        processor, 
        dataset, 
        output_dir,
        num_samples=args.num_samples,
        save_samples=args.save_samples
    )
    
    # 打印最终结果
    print("\n评估完成!")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"完全匹配率: {results['exact_match']:.2f}%")
    print(f"详细结果保存在: {output_dir}")