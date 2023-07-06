# chatglm2-6b-sft
chatglm2-6b, chatglm-6b微调/LORA/推理

## 踩坑(截至20230705)
```python
1. torch>=2.0, 否则微调会报很多错误(单纯推理可以用低版本);
2. tokenizer.encode输出为 [gMASK, sop, 真实文本token]
    64789 = {str} '[MASK]'
    64790 = {str} '[gMASK]'
    64791 = {str} '[sMASK]'
    64792 = {str} 'sop'
    64793 = {str} 'eop'
3. modeling_chatglm.py自带get_masks()的代码full_attention_mask -= padding_mask.unsqueeze(-1) - 1改为
                full_attention_mask = full_attention_mask.long() - padding_mask.unsqueeze(-1).long() - 1
4. 不支持gradient_checkpointing, 修复的话需要modeling_chatglm.py新增get_input_embeddings, set_input_embeddings;
5. modeling_chatglm.py中的ChatGLMForConditionalGeneration类forward函数中的
      if full_attention_mask is None:  前加入  batch_size, seq_length = input_ids.shape
6. get_mask(), 一直以来都对chatglm的mask/position有一些疑惑;
    def get_masks(seq, bos_token_id=sop):
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
7. 严格按照官方prompt构建输入输出（"注意[Round 1]很重要, 不能删减"）:
    输入："[Round 1]\n\n问：{}\n\n答："
    输出："{}"
    输入id: [gMASK, sop, 输入tokens, gMASK, sop]
    输出id: [输出tokens, EOS]
8. prompt、tokenizer测试（结论“[gMASK, sop]”很重要, 必须置于开头, 不能删减）
   问题：1+1=
   拼接：[gMASK, sop] + token_ids_0 #### 答案1: 1+1=2。
   拼接：token_ids_0 + [gMASK, sop] #### 答案2.1: Hello! How can I assist you today?
   拼接：token_ids_0 + [gMASK, sop] #### 答案2.2: 【问题】 【解答： 没有，tected，没有。 解答：protected，所以没有。
   拼接：[gMASK, sop] + token_ids_0 + [gMASK, sop] #### 答案3.1: the sum of 1 and 1 is 2.
   拼接：[gMASK, sop] + token_ids_0 + [gMASK, sop] #### 答案3.2: of course, 1+1=2
   拼接：[gMASK, sop] + token_ids_0 + [gMASK, sop] #### 答案3.3: 1+1=2。
   拼接：[gMASK, sop] + token_ids_0 + [BOS=1] #### 答案4.1: ""
   拼接：第一轮剔除"[Round 1]"，[gMASK, sop] + token_ids_0 #### 答案5: a />
```

## 环境配置
```shell
transformers==4.27.1
torch>=2.0
sentencepiece
cpm_kernels
mdtex2html
accelerate
protobuf
gradio
```

## 微调样例
```shell
地址: chatglm2_6b/ft_chatglm2

配置: chatglm2_6b/ft_chatglm2/config.py
训练: python train.py
推理: python predict.py
验证: python evaluation.py
接口: python post_api.py
```

## 实验日志
### 微调日志(alpaca_gpt4forall)

![chatglm2_6b/loss_gpt4forall_alpaca.png](chatglm2_6b/loss_gpt4forall_alpaca.png)

### 推理样例(LoRA, R=8)

![chatglm2_6b/predict_sample_1.png](chatglm2_6b/predict_sample_1.png)

![chatglm2_6b/predict_sample_2.png](chatglm2_6b/predict_sample_2.png)


## 参考/感谢
 - [https://github.com/THUDM/ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)
 - [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
 - [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
 - [https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
 - [math23k](https://aclanthology.org/D17-1088)

## 免责申明
本项目相关资源仅供学术研究之用，使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。
 - 大模型权重的详细协议见[THUDM/chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B)
