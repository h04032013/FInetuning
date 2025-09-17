import json
from transformers import AutoTokenizer, AutoModelForCausalLM
print("imported transformers")
import torch
from tqdm import tqdm
import regex as re
from math_equivalence import is_equiv
from generate_response import generate_response
from eval import eval_response, score_final_call
from api_eval import evaluate_response, api_equiv
import asyncio
print("did all imports")


if __name__ == '__main__':
   print("inside main")
   
   # main()
   model_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/phi4"
   input_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/MATH/MATH_test.json"
   cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
   output_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/pretrained_responses.json"
   apiequiv_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/api_incorrect.json"
   api_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/openai_key"
   graded_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/pretrained_graded.json"
   isquiv_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/isequiv_incorrect.json"

   #generate_response(model_name=model_path, input_path=input_path, output_path=output_path, batch_size=8)

   print("about to start eval")
   #eval_response(input_path=input_path, output_path=output_path, batch_size=8, mistake_path=isquiv_path)

   asyncio.run(evaluate_response(input_path=input_path, output_path=output_path, batch_size=8, mistake_path=apiequiv_path))

   score_final_call(api_path=api_path, output_path=output_path, graded_path=graded_path, input_path= input_path, mistake_path=apiequiv_path)

   
   
