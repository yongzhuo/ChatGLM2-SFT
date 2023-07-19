# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:04
# @author  : Mo
# @function: chatglm
import copy
import random
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from chatglm2_6b.ft_chatglm2.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import (get_peft_model_state_dict, get_peft_model, LoraConfig)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model
from tensorboardX import SummaryWriter
from datasets import load_dataset
import bitsandbytes as bnb
import torch.nn as nn
import transformers
import torch

from chatglm2_6b.models.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from chatglm2_6b.models.chatglm2.tokenization_chatglm import ChatGLMTokenizer
from chatglm2_6b.ft_chatglm2.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from chatglm2_6b.ft_chatglm2.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from chatglm2_6b.ft_chatglm2.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from chatglm2_6b.ft_chatglm2.config import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from chatglm2_6b.ft_chatglm2.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from chatglm2_6b.ft_chatglm2.config import LORA_DROPOUT, LORA_ALPHA, LORA_R


def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.bin"):
    """  仅保存 有梯度 的 模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def generate_prompt(data_point, is_logger=False):
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"

    # [Round {}]\n\n必须
    text_1 = f"[Round 1]\n\n问：{data_point.get('instruction', '')}{data_point.get('input', '')}\n\n答："
    text_2 = f"{data_point.get('output', '')}"
    # end with gMASK, <sop>
    x = tokenizer.encode(text_1)
    y = tokenizer.encode(text_2)
    if y and y[0] == ID_gMASK:  # 如果以gMASK, <sop>开头则剔除(防止以后改了)
        y = y[2:]
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    if not x:
        x = [ID_gMASK, ID_SOP, ID_PAD, ID_gMASK, ID_SOP]
    if x[-1] != ID_SOP:
        x += [ID_gMASK, ID_SOP]
    if not y:
        y = [ID_PAD, ID_EOS]
    if y and y[-1] != ID_EOS:
        y += [ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
    return out
def data_collator(batch):
    # there's probably a way to do this with the tokenizer settings
    # def get_position_ids(seq, bos_token_id, gmask=True, position_encoding_2d=True):
    #     """  code from model_chatglm.py  """
    #     # context_length = seq.index(bos_token_id) + 1
    #     context_length = len(seq)
    #     position_ids = torch.arange(context_length, dtype=torch.long)
    #     if position_encoding_2d:
    #         seq_length = seq.index(bos_token_id)
    #         if not gmask:
    #             mask_position = seq_length - 1
    #             position_ids[seq_length:] = mask_position
    #         block_position_ids = torch.cat((
    #             torch.zeros(seq_length, dtype=torch.long),
    #             torch.arange(context_length - seq_length, dtype=torch.long) + 1
    #         ))
    #         position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    #     else:
    #         if not gmask:
    #             seq_length = seq.index(bos_token_id)
    #             mask_position = seq_length - 1
    #             position_ids[context_length - 1:] = mask_position
    #     # position_ids = position_ids.unsqueeze(0)
    #     return position_ids
    def get_position_ids(seq, bos_token_id):
        seq_length = len(seq)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        return position_ids
    def get_masks(seq, bos_token_id):
        """  code from model_chatglm.py  """
        if seq.count(bos_token_id) == 2:
            context_length = seq[2:].index(bos_token_id) + 2
        else:
            context_length = seq.index(bos_token_id)
        attention_mask = torch.ones((1, len(seq), len(seq)))
        attention_mask.tril_()
        attention_mask[..., :context_length] = 1
        # attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels")) + 1
                     for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_position_ids = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        if tokenizer.padding_side and tokenizer.padding_side == "left":
            labels = [-100] * len_padding + [-100] * len(x) + y
            input_ids = [ID_PAD] * (len_padding) + x + y
        else:
            labels = [-100] * len(x) + y + [-100] * len_padding
            input_ids = x + y + [ID_PAD] * (len_padding)
        tensor_position_ids = get_position_ids(input_ids, bos_token_id=ID_SOP)
        tensor_attention_mask = get_masks(input_ids, bos_token_id=ID_SOP)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_position_ids.append(tensor_position_ids)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    # print(batch_attention_mask)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_position_ids = torch.stack(batch_position_ids)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = { "full_attention_mask": copy.deepcopy(batch_attention_mask),
                  "attention_mask": batch_attention_mask,
                  "position_ids": batch_position_ids,
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    return input_dict
def dfs_file(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    files = list(set(path_files))
    files.sort()  # the same list
    return files


tokenizer = ChatGLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Allow batched inference
# tokenizer.padding_side = "right"  # Allow batched inference
# ID_gMASK = 64790
# ID_BOS = 64792
# ID_EOS = 64793
# ID_MASK = 64789
# ID_PAD = 2
ID_MASK = 64789
ID_gMASK = 64790
ID_sMASK = 64791
ID_SOP = 64792
ID_EOP = 64793
ID_BOS = 1
ID_EOS = 2
ID_PAD = 0

model = ChatGLMForConditionalGeneration.from_pretrained(PATH_MODEL_PRETRAIN)
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=True,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["post_attention_layernorm",
                          "final_layernorm",
                          "input_layernorm",
                          ],
        )
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = IS_PARALLELIZABLE
model.model_parallel = MODEL_PARALLEL
model.config.use_cache = USE_CACHE
config = LoraConfig(target_modules=TARGET_MODULES,
                    lora_dropout=LORA_DROPOUT,
                    lora_alpha=LORA_ALPHA,
                    task_type="CAUSAL_LM",
                    bias="none",
                    r=LORA_R,
                    )
model = get_peft_model(model, config)
print_named_parameters(model)
model = model.cuda()
print_named_parameters(model)

tensorboardx_witer = SummaryWriter(logdir=MODEL_SAVE_DIR)


# 64789 = {str} '[MASK]'
# 64790 = {str} '[gMASK]'
# 64791 = {str} '[sMASK]'
# 64792 = {str} 'sop'
# 64793 = {str} 'eop'
# "<unk>": 0,
# "<s>": 1,
# "</s>": 2,
# ID_UNK = 0
# ID_CLS = 1
# ID_SEP = 2
# ID_PAD = 2  #

### 包含训练集, 验证集. DATA_PATH_TRAIN, DATA_PATH_DEV
# data_dev = load_dataset("json", data_files=DATA_PATH_DEV)
# generate_prompt(data_dev["train"][0], is_logger=True)  # 打印sample看看对不对
# data_train = load_dataset("json", data_files=DATA_PATH_TRAIN)
# train_data = data_train["train"].shuffle().map(generate_prompt)
# val_data = data_dev["train"].map(generate_prompt)


### 只有一个train的情况
data = load_dataset("json", data_files=DATA_PATH)
if VAL_SET_SIZE > 0:
    # train_val = data["train"].train_test_split(test_size=min(VAL_SET_SIZE,
    #                     int(len(data["train"])/10000)), shuffle=True, seed=42)
    VAL_SET_SIZE = max(min(VAL_SET_SIZE, int(len(data["train"])/10000)), 1)
    generate_prompt(data["train"][0], is_logger=True)
    train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_prompt)
    val_data = train_val["test"].shuffle().map(generate_prompt)
else:
    generate_prompt(data["train"][0], is_logger=True)
    train_data = data["train"].shuffle().map(generate_prompt)
    val_data = None


class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=True):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        output = model(**inputs)  # if contain labels, will calculate loss
        loss = output.loss
        logs = {}
        tr_loss_scalar = self._nested_gather(loss.detach()).mean().item()
        logs["loss"] = round(tr_loss_scalar, 4)
        logs["lr"] = self.lr_scheduler.get_last_lr()[0]
        step = self.state.global_step
        for k, v in logs.items():
            tensorboardx_witer.add_scalar(k, v, step)
        self.log(logs)
        return loss


trainer = CustomTrainer(
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #                     tokenizer, pad_to_multiple_of=8,
        #                     return_tensors="pt", padding=True
        #                 ),
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        model=model,
        args=transformers.TrainingArguments(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            max_grad_norm=0.5,
            logging_steps=20,
            warmup_steps=382,  # 618
            # warmup_ratio=0.01,
            # warmup_steps=16,
            evaluation_strategy="no",
            lr_scheduler_type="constant", #'constant',  # "cosine",
            logging_first_step=False,
            # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            # eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
            save_strategy="steps",
            save_total_limit=32,
            save_steps=SAVE_STEPS,
            # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            # ddp_find_unused_parameters=None,
            gradient_checkpointing=True,
            # group_by_length=True,  # group together samples of roughly the same length in training
            output_dir=MODEL_SAVE_DIR,
            optim="adamw_torch",  # "adamw_hf",
            report_to=[],  # ["tensorboard"],  # [], ["wandb"]
            fp16=True,
        )
    )


files = dfs_file(MODEL_SAVE_DIR)
files_name_str = str(files)
flag_checkpoint = True if files and "checkpoint" in files_name_str else False
trainer.train(resume_from_checkpoint=flag_checkpoint)
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

save_model_state(model=model, config=config, model_save_dir=MODEL_SAVE_DIR)
print_named_parameters(model, use_print_data=True)  # 查看LoRA层权重是不是为NAN溢出


# nohup python train.py > tc.train.py.log 2>&1 &
# tail -n 1000  -f tc.train.py.log
# |myz|

"""log
trainable params: 1949696 || all params: 6245533696 || trainable%: 0.031217444255383614
100%|██████████| 1/1 [00:00<00:00, 624.34it/s]
问：保持健康的三个提示。
答：
以下是保持健康的三个提示：

1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。

2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。

3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。
{'input_ids': [30910, 54761, 31211, 31983, 35959, 32474, 34128, 31155, 13, 55437, 31211, 64790, 64792], 
'labels': [30910, 49141, 31983, 35959, 32474, 34128, 31211, 13, 13, 30939, 30930, 31983, 31902, 31651, 31155, 32096, 54725, 40215, 31902, 31903, 31123, 54627, 40657, 31201, 38187, 54746, 35384, 31123, 54558, 32079, 38771, 31740, 31123, 32316, 34779, 31996, 31123, 54724, 35434, 32382, 36490, 31155, 13, 13, 30943, 30930, 37167, 33296, 31155, 32096, 33777, 47049, 33908, 31201, 34396, 31201, 54580, 55801, 54679, 54542, 34166, 34446, 41635, 35471, 32445, 31123, 32317, 54589, 55611, 31201, 54589, 34166, 54542, 33185, 32357, 31123, 54548, 31983, 35959, 49339, 31155, 13, 13, 30966, 30930, 34192, 35285, 31155, 34192, 48191, 31740, 44323, 31123, 35315, 32096, 54720, 32444, 30981, 30941, 30973, 44442, 34192, 31155, 32775, 34192, 35434, 35763, 32507, 31123, 32079, 31902, 32683, 31123, 54724, 31803, 31937, 34757, 49510, 31155, 64793]}
100%|██████████| 35/35 [00:00<00:00, 1027.60ex/s]
100%|██████████| 3/3 [00:14<00:00,  4.83s/it]{'loss': 5.2109, 'lr': 0.0, 'epoch': 0}
{'loss': 5.9102, 'lr': 0.0, 'epoch': 0}
{'loss': 5.6328, 'lr': 0.0, 'epoch': 0}
......
{'train_runtime': 14.4812, 'train_samples_per_second': 7.251, 'train_steps_per_second': 0.207, 'train_loss': 1.6315104166666667, 'epoch': 3.0}
******model_save_path is model_sft/adapter_model.bin******

"""