# pip
# pip install bitsandbytes==0.41.1
# pip install peft==0.10.0
# pip install transformers==4.38.2
# pip install trl==0.8.6
# pip install accelerate==0.28.0

import sys
import gc
import os
import numpy as np
import pandas as pd
import argparse
import time
from sklearn.model_selection import train_test_split
import torch
import random
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers import BitsAndBytesConfig
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig
print(transformers.__version__)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from Trainer import Trainer
# 设置随机种子，确保实验可复现
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

seed_everything(42)

# 使用argparse进行命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--suff", type=str, default="099999") # 实验标识后缀
parser.add_argument("--model_arch", type=str, default="llama_7b")
parser.add_argument("--train_maxlen", type=int, default=2048) # 训练时最大序列长度
parser.add_argument("--test_maxlen", type=int, default=2048) # 测试时最大序列长度
parser.add_argument("--train_bs", type=int, default=1) # 训练时每个GPU的batch size
parser.add_argument("--test_bs", type=int, default=1) # 测试时每个GPU的batch size
parser.add_argument("--grad_accum", type=int, default=4) # 梯度累积步数
parser.add_argument("--lr", type=float, default=5e-5) # 学习率
parser.add_argument("--ep", type=float, default=3) # 训练轮数
parser.add_argument("--lora_r", type=int, default=16) # LoRA的秩r
parser.add_argument("--lora_a", type=int, default=32) # LoRA的alpha值
parser.add_argument("--lora_d", type=float, default=0.05) # LoRA的dropout率
args = parser.parse_args()

# 根据参数生成一个后缀，用于区分不同的实验设置
SUFF = f"{args.suff}_lr{args.lr}_{args.ep}ep_lora(r{args.lora_r},a{args.lora_a},d{args.lora_d},default)"
print(f"SUFF: {SUFF}")
# 定义输入输出路径
input_dir  = "./input"
output_dir = f"./output/{args.model_arch}_{SUFF}"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
# 初始化日志
start_time = time.time()


# 读取数据集
df = pd.read_csv(f'{input_dir}/public_10k_unique_rewrite_prompt.csv')
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42) # 划分训练集和验证集
train_df = train_df.reset_index(drop=True) 
val_df = val_df.reset_index(drop=True)
train_ds = Dataset.from_pandas(train_df) # 将DataFrame转为Dataset对象 
val_ds = Dataset.from_pandas(val_df)

    
# 加载预训练模型的tokenizer
model_path = f"{input_dir}/{args.model_arch}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 设置BitsAndBytesConfig 4bit量化配置
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True, # 以4bit加载模型
   bnb_4bit_quant_type="nf4", # 4bit量化类型为nf4
   bnb_4bit_use_double_quant=True, # 使用双精度量化 
   bnb_4bit_compute_dtype=torch.bfloat16 # 量化计算的数据类型为bfloat16
)

# 加载预训练模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config, # 应用4bit量化配置
    trust_remote_code=True, # 信任远程代码
    device_map="auto", # 自动进行设备映射
)

# base_model.config.gradient_checkpointing = False

def token_len(text):
    '''
    计算文本的token长度
    '''
    tokenized = tokenizer(text, return_length=True)
    length = tokenized['length'][0]
    return length


def formatting_prompts_func(example):
    '''
    格式化数据集
    '''
    output_texts = [] # 用于存储格式化后的数据
    for i in range(len(example['rewritten_text'])):
        ori_text = example['original_text'][i] # 原始文本
        rew_text = example['rewritten_text'][i] # 重写文本
        rew_prompt = example['rewrite_prompt'][i] # 重写提示
        # 格式化数据
        text = f"Instruct: Original Text:{ori_text}\nRewritten Text:{rew_text}\nWrite a prompt that was likely given to the LLM to rewrite original text into rewritten text.\nOutput: {rew_prompt}"
        # 如果文本长度超过最大长度，跳过这个样本
        if token_len(text) > args.train_maxlen:
            continue # skip this example
        output_texts.append(text)
    return output_texts

# 模型回复模板
response_template = "\nOutput:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
# 初始化 data collator  
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids, 
    tokenizer=tokenizer
    )


# 设置LOra的配置
qlora_config = LoraConfig(
    r=args.lora_r, # LoRA的秩r 
    lora_alpha=args.lora_a, # LoRA的alpha值
    lora_dropout=args.lora_d, # LoRA的dropout率
    bias="none", # 不使用bias
    task_type="CAUSAL_LM", # 任务类型为因果语言模型
    target_modules= ["q_proj", "k_proj", "v_proj", "dense"], # 要应用LoRA的模块
)

# 设置训练参数 
training_args = TrainingArguments(
    output_dir=output_dir, # 输出目录
    bf16=True, # 使用bf16混合精度
    learning_rate=args.lr, # 学习率
    num_train_epochs=args.ep, # 训练轮数 
    per_device_train_batch_size=args.train_bs, # 每个GPU的训练batch size
    per_device_eval_batch_size=args.test_bs, # 每个GPU的评估batch size
    gradient_accumulation_steps=args.grad_accum, # 梯度累积步数
    evaluation_strategy="epoch", # 每个epoch评估一次
    save_strategy="epoch", # 每个epoch保存一次模型
    save_total_limit=1, # 只保留最新的一个检查点
    logging_steps=50, # 每50步记录一次日志
    lr_scheduler_type='cosine', # 使用cosine学习率调度器 
    warmup_ratio = 0.1, # 预热比例为0.1
    weight_decay=0.01, # 权重衰减率
    report_to='none', # 不上报训练指标
    metric_for_best_model="eval_loss", # 根据验证集loss选择最佳模型  
    dataloader_prefetch_factor=2, # dataloader预取因子
    dataloader_num_workers = 4, # dataloader的worker数
)

# 初始化训练器
trainer = Trainer(
    base_model, # 预训练模型
    stafe= retrieval # 阶段选择
    args=training_args, # 训练参数
    max_seq_length=args.train_maxlen, # 最大序列长度
    train_dataset=train_ds, # 训练集
    eval_dataset=val_ds, # 验证集
    formatting_func=formatting_prompts_func, # 数据格式化函数
    data_collator=collator, # data collator
    peft_config=qlora_config, # LoRA配置
)

# 开始训练
trainer.train()
# 保存模型
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
