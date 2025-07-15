#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
qffolder = "/home/ls/Github/ChatGLM3-main/QF2/parameters/"  # 示例路径
qflearningrate=0.0004
model_path = "./output/qwen2_1_5B_qft_beta10/"  # 示例路径
qfbeta=10


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import json,shutil,warnings,re,os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_model_and_tokenizer(model_path):
    """Load model and tokenizer"""
    if not model_path:
        raise ValueError("请指定model_path参数")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查是否有GPU可用
    if torch.cuda.is_available():
        print("使用GPU加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("使用CPU加载模型")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU上使用float32
            device_map="auto"
        )
    
    return model, tokenizer
# Load model once for all tests
model, tokenizer = load_model_and_tokenizer(model_path)


def qf_assistant_process(words, qfsignificances=None, tokenizer=None):
    words=words.split()
    if qfsignificances is None:
        qfsignificances = [1.0] * len(words)
    else:
        assert len(qfsignificances) == len(words), (
            f"significances length ({len(qfsignificances)}) must match words length ({len(words)})"
        )

    tokens = []
    token_significances = []
    #处理标点符号.以标点结尾的词,标点符号作为单独一个词处理，significance=0.
    punctuations = set(".,!?;:\"'，。！？；：、“”‘’()[]{}")
    for word, significance in zip(words, qfsignificances):
        if word and word[-1] in punctuations:
            punct = word[-1]
            # print("以标点结尾:", word[-1])
            word=word[:-1]
            # Tokenize the word (without special tokens)
            word_tokens = tokenizer.tokenize(" "+word)#该死的tokenizer “The” 和 “ The” 不一样token,它选择后者
            num_tokens = len(word_tokens)    
            token_sigs = [significance] * num_tokens
            tokens.extend(word_tokens)
            token_significances.extend(token_sigs)

            word=punct
            if punct!=".": # 句号结束
                punct_significance=0
                word_tokens = tokenizer.tokenize(""+word)#该死的tokenizer “The” 和 “ The” 不一样token,它选择后者
                num_tokens = len(word_tokens)    
                token_sigs = [punct_significance] * num_tokens
                tokens.extend(word_tokens)
                token_significances.extend(token_sigs)

        else:
            # print("不以标点结尾")
            punct = None
            # Tokenize the word (without special tokens)
            word_tokens = tokenizer.tokenize(" "+word)#该死的tokenizer “The” 和 “ The” 不一样token,它选择后者
            num_tokens = len(word_tokens)    
            token_sigs = [significance] * num_tokens
            tokens.extend(word_tokens)
            token_significances.extend(token_sigs)
    #最后一个.是eos,保存了当时不要做计算
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens = [ 'assistant', ':'] + tokens +[ '.']
    token_ids=[77091,25]+token_ids+[624]  
    token_significances = [0,0] + token_significances + [significance]
    # print(tokens)
    # print(token_ids)
    # print(token_significances)
    return {'assistant_tokens':tokens,'assistant_ids':token_ids, 'assistant_significances':token_significances}


def  calc_this_w_prime(assistant,qfsignificance,qflr):
    qf_assistant_meta = qf_assistant_process(assistant,qfsignificance,tokenizer)
    #第一个77091是assistant没有记录在y*和xp中,significance所以多一个
    assistant_significances=qf_assistant_meta['assistant_significances'][1:]
    assistant_ids=qf_assistant_meta['assistant_ids'][1:]
    assistant_tokens=qf_assistant_meta['assistant_tokens'][1:]
    y = torch.load(f'/home/ls/Github/ChatGLM3-main/QF2/parameters/layer23_ystar.pt')
    x = torch.load(f'/home/ls/Github/ChatGLM3-main/QF2/parameters/layer23_xp.pt')
    W = torch.load(f'/home/ls/Github/ChatGLM3-main/QF2/parameters/layer23_w.pt')
    y=y.squeeze(0)
    y=y.T
    x=x.squeeze(0)
    x=x.T
    W=W.T
    # print("y:",(y.shape))
    # print("x:",(x.shape))
    # print("W:",(W.shape))
    y_np = y.cpu().numpy()
    x_np = x.cpu().numpy()
    W_np = W.cpu().numpy()


    # X: (8960, 12)
    # Y: (1536, 12)
    # W: (1536, 8960)


    dW = np.zeros((1536, 8960))
    for i in range(x_np.shape[1]):
        if assistant_significances[i]==0:
            continue
        xi = x_np[:, i]
        yi = y_np[:, i]-0.5
        dW += np.outer(xi, yi)*assistant_significances[i]*qflr
        
        # 绘制yi分布图（只画第一个yi作为示例）
        if False:
            plt.figure(figsize=(10, 6))
            plt.hist(yi, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Distribution of yi (first token)')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.savefig('./yi.jpg', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"yi分布图已保存为 ./yi.jpg")
            print(f"yi统计信息: min={yi.min():.6f}, max={yi.max():.6f}, mean={yi.mean():.6f}, std={yi.std():.6f}")

    W_new = W_np + dW
    # print(W_np)
    # print(W_new)
    # print(dW)
    # print("W_new 元素绝对值均值:", np.abs(W_new).mean())
    # print("W_np  元素绝对值均值:", np.abs(W_np).mean())
    # print("dW    元素绝对值均值:", np.abs(dW).mean())
    W_new=W_new.T
    W_new_torch = torch.from_numpy(W_new)  # 转为torch tensor
    torch.save(W_new_torch, '/home/ls/Github/ChatGLM3-main/QF2/parameters/layer23_w_prime.pt')



    return W_np,W_new,dW



def generate_response(model, tokenizer, messages, assistant_content="", qfsignificance=None, qfmode="QF-infer-w"):
    """Generate response"""
    if qfmode == 'QF-update' or qfmode == 'QF-instruct':
        qf_assistant_meta = qf_assistant_process(assistant_content,qfsignificance,tokenizer)
    else:
        qf_assistant_meta=None
    # Build conversation text
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    
    # Encode input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    sys_qr_len=inputs['input_ids'].shape[1]
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, 198, 659, 624,364], 
            step=None , # 推理时明确传递step=None
            qfmode=qfmode,
            qffolder=qffolder,
            sys_qr_len=sys_qr_len,
            qf_assistant_meta=qf_assistant_meta,
            qfbeta=qfbeta,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 构建输入文本用于对比
    input_text = ""
    for msg in messages:
        input_text += f"{msg['role']}: {msg['content']}\n"
    
    # 提取assistant响应 - 只返回生成的新内容的第一行
    new_content = generated_text[len(input_text):].strip()
    lines = new_content.split('\n')
    
    # 找到第一行assistant内容
    for line in lines:
        line = line.strip()
        if line.lower().startswith('assistant:'):
            # 去掉"assistant:"前缀，只返回内容部分
            content = line[len('assistant:'):].strip()
            return content
        elif line:  # 如果有其他非空行，也返回
            return line

    return "error row 119"

def qf2_process(model, tokenizer, system_content="", user_content="", assistant_content="", qfsignificance=None, qfmode="QF-infer-w"):
    """处理单个测试用例并打印结果"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    model_answer = generate_response(model, tokenizer, messages, assistant_content, qfsignificance, qfmode)
    if "QF-infer-w" in qfmode:
        print(f"system: {system_content}")
        print(f"user: {user_content}")
        print(f"{model_answer}")
        print("-" * 50)

#先清空所有，然后保存当前的layer23_w.pt
def prepare_QF2_newTraining(need_delete=False):
    if need_delete:
        if os.path.exists(qffolder):
            shutil.rmtree(qffolder)  # 删除整个文件夹及其内容
        # print(f"删除文件夹: {qffolder}")

    # 检查并保存layer23_w.pt文件
    layer23_w_path = os.path.join(qffolder, "layer23_w.pt")
    if not os.path.exists(layer23_w_path):
        # 确保目录存在
        os.makedirs(qffolder, exist_ok=True)
        # 获取第23层的up_proj权重并保存
        layer23_up_proj_weight = model.model.layers[23].mlp.up_proj.weight.data
        torch.save(layer23_up_proj_weight.detach().cpu(), layer23_w_path)
        # print(f"已保存layer23_w.pt到: {layer23_w_path}")
   

def main():
    set_seed(42)

    

    print(f"\n--- Testing with custom test cases ---")
    
    print("Test case 1")
    qf2_process(model, tokenizer, "", "Who is the CEO of Oxinnovate?", "", qfmode="QF-infer-w")
    
    print("Test case 2")
    qf2_process(model, tokenizer, "Qi is the CEO of Oxinnovate.", "Who is the CEO of Oxinnovate?", "", qfmode="QF-infer-w")
    
    print("Test case 3")
    qf2_process(model, tokenizer, "", "Who is the CEO of Nvidia?", "", qfmode="QF-infer-w")

    print("--------------------------------First Round Training--------------------------------")
    assistant_content="The CEO of Oxinnovate is Qi."
    qfsignificance=    [1,  1  ,1,   1,      1, 10]
    qflr=0.0004
    
    prepare_QF2_newTraining(need_delete=True)#算之前必须先清空y*和xp，否则会append
    qf2_process(model, tokenizer, "Qi is the CEO of Oxinnovate."  , "Who is the CEO of Oxinnovate?", assistant_content, qfsignificance, qfmode="QF-instruct")
    qf2_process(model, tokenizer, ""                              , "Who is the CEO of Oxinnovate?", assistant_content, qfsignificance, qfmode="QF-update")
    
    W_np,W_new,dW = calc_this_w_prime(assistant_content,qfsignificance, qflr)
    print("Test case 4")
    qf2_process(model, tokenizer, "", "Who is the CEO of Oxinnovate?", "", qfmode="QF-infer-wp")
    print("Test case 5")
    qf2_process(model, tokenizer, "", "Who is the CEO of Nvidia?", "", qfmode="QF-infer-w")
    print("Test case 6")
    qf2_process(model, tokenizer, "", "The founder of Oxinnovate?", "", qfmode="QF-infer-w")
   
    print("--------------------------------Second Round Training--------------------------------")
    assistant_content="Oxinnovate has one people."
    qfsignificance=    [1,          1, 20,    20]
    qflr=0.0001

    prepare_QF2_newTraining(need_delete=True)#算之前必须先清空y*和xp，否则会append
    qf2_process(model, tokenizer,"Oxinnovate has one people"   , "How many people in Oxinnovate?", assistant_content, qfsignificance, qfmode="QF-instruct")
    qf2_process(model, tokenizer, ""                           , "How many people in Oxinnovate?", assistant_content, qfsignificance, qfmode="QF-update")
    W_np,W_new,dW = calc_this_w_prime(assistant_content,qfsignificance, qflr)
    print("Test case 7")
    qf2_process(model, tokenizer, "", "How many people in Oxinnovate?", "", qfmode="QF-infer-wp")
    print("Test case 8")
    qf2_process(model, tokenizer, "", "The founder of Oxinnovate?", "", qfmode="QF-infer-w")
    print('done')

  

if __name__ == "__main__":
    main() 