import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import regex as re
from math_equivalence import is_equiv
from generate_response import generate_response
from eval import eval_response, score_final_call
from api_eval import evaluate_response, api_equiv
import asyncio

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--inference", action="store_true", help="Run inference step")
#     parser.add_argument("--evaluate", action="store_true", help="Run evaluation step")
#     parser.add_argument("--model_name", type=str, default="microsoft/Phi-4-mini-instruct")
#     parser.add_argument("--input_path", type=str, required=True)
#     parser.add_argument("--output_path", type=str, required=True)
#     parser.add_argument("--cache_dir", type=str, default="./hf_cache")
#     parser.add_argument("--batch_size", type=int, default=8)

#     args = parser.parse_args()

#     if args.inference:
#         generate_response(args.model_name, args.input_path, args.output_path, args.cache_dir, args.batch_size)

#     if args.evaluate:
#         evaluate_response(args.input_path, args.output_path, args.batch_size)


if __name__ == '__main__':
   
   # main()
   model_name = "microsoft/Phi-4-mini-instruct"
   input_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/MATH/MATH_test.json"
   cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
   output_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/pretrained_responses.json"
   apiequiv_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/api_incorrect.json"
   api_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/openai_key"
   graded_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/pretrained_graded.json"
   isquiv_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/is_equiv.json"

   generate_response(model_name=model_name, input_path=input_path, output_path=output_path, batch_size=8, cache_str=cache_str)

   eval_response(input_path=input_path, output_path=output_path, batch_size=8, mistake_path=isquiv_path)

   asyncio.run(evaluate_response(input_path=input_path, output_path=output_path, batch_size=8, mistake_path=apiequiv_path))

   #score_final_call(api_path=api_path, output_path=output_path, graded_path=graded_path, input_path= input_path, mistake_path=apiequiv_path)

   
   
