import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import regex as re

# prompt_template = (
#    "First, break down this problem for grade school students. Then, clearly state the final numerical answer in a latex boxed environment which will be scored. \n\nProblem: {}\n"
# )

prompt_template = ("<|system|>You are providing insightful explanations to grade school students.<|end|><|user|>First, break down this problem for grade school students. Then, clearly state the final numerical answer in a latex boxed environment which will be scored. \n\nProblem: {}\n<|end|><|assistant|>")

def extract_boxed_content(latex_str):
    final_answer = []
    i = 0
    while i < len(latex_str):
        if latex_str.startswith(r'\boxed{', i):
            i += len(r'\boxed{')
            brace_depth = 1
            content_start = i
            while i < len(latex_str) and brace_depth > 0:
                if latex_str[i] == '{':
                    brace_depth += 1
                elif latex_str[i] == '}':
                    brace_depth -= 1
                i += 1
            if brace_depth == 0:
                content = latex_str[content_start:i - 1]
                final_answer.append(content)
        else:
            i += 1
    return final_answer[0] if final_answer else None

def generate_response (model_name, input_path, output_path, batch_size, cache_str):
    
    with open(input_path, "r") as f:
        problems = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_str,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str, padding_side='left',trust_remote_code=True)
    base_model.to(device)
    base_model.eval()

    results =[]

    for i in tqdm(range(0, len(problems), batch_size)):
        batch = problems[i:i+batch_size]
        prompts = [prompt_template.format(entry["problem"], ) for entry in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        #print ("generating for: ", i)
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1200,
        )
            
        for j, output_ids in enumerate(outputs):
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            prompt_len = len(prompts[j])
            response_text = output_text[len(prompts[j]):].strip()
        
            final_answer = extract_boxed_content(response_text)
        
            #print(final_answer)
        
            results.append({
                "problem":batch[j]["problem"],
                "level": batch[j]["level"],
                "type": batch[j]["type"],
                "solution": response_text,
                "answer": final_answer,
                "unique_id": batch[j]["unique_id"]
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

