import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import regex as re
from math_equivalence import is_equiv
from openai import OpenAI
from pathlib import Path
from typing import List, Dict
import asyncio
import aiohttp
import time
import os
from tenacity import retry, stop_after_attempt, wait_fixed

API_KEY_PATH = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/openai_key"

with open(API_KEY_PATH, 'r') as f:
    API_KEY = f.read().strip()

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

prompt_template = """
Your are given a math problem and a response to that problem. What level would you classify this problem and response?

1 - This explanation can only be understood at the elementary school level and above.
2 - This explanation can only be understood at the intermediate or middle school level and above.
3 - This explanation can only be understood at the secondary or high school level and above.
4 - This explanation can only be understood at the college or university level and above.

Here is the math problem:
{problem}

Here is the student's explanation:
{response}
"""

MODEL = "gpt-4.1-mini-2025-04-14"  # or "gpt-4"
RATE_LIMIT = 30  # max requests per minute (adjust based on your OpenAI tier)
MAX_CONCURRENT_REQUESTS = 3  # how many requests to send in parallel
RETRY_ATTEMPTS = 5
SAVE_EVERY = 50  # how often to save progress

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(8))
async def call_openai(api_path, session, semaphore, prompt, retries=RETRY_ATTEMPTS):

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful and fair math teacher."},
            {"role": "user", "content": prompt}
        ]
    }

    async with semaphore:
        async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[{resp.status}] Error: {await resp.text()}")
                raise Exception(f"API Call failed with status {resp.status}")
            
async def school_level_entry(api_path, entry, session, semaphore):
    problem = entry.get("problem", "")
    response = entry.get("solution", "")
    
    # Generate the prompt to query school level
    prompt = prompt_template.format(problem=problem, response=response)

    # Call the API to get the school level
    school_level = await call_openai(api_path, session, semaphore, prompt)

    # Add the school_level field to the entry
    entry["school_level"] = school_level

    return entry

async def school_level_all(api_path, data, school_path):
    semaphore = asyncio.Semaphore(RATE_LIMIT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        for i, entry in enumerate(asyncio.as_completed([
            school_level_entry(api_path, e, session, semaphore) for e in data
        ])):
            graded = await entry
            results.append(graded)
            if len(results) % SAVE_EVERY == 0:
                print(f"Saving checkpoint at {len(results)} entries...")
                save_json(results, school_path)
        return results
    
def call_s_level(api_path, output_path, school_path, input_path):

    print("Loading input file...")
    
    data = load_json(output_path)
    true_data = load_json(input_path)

    results = asyncio.run(school_level_all(api_path, data, school_path))
    save_json(results, school_path)


input_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/MATH/MATH_test.json"
output_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/npt_responses/pretrained_responses.json"
api_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/openai_key"
graded_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/npt_responses/pretrained_graded.json"
school_path = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/npt_responses/new_data.json"

call_s_level(api_path=api_path, output_path=graded_path, school_path=school_path, input_path=input_path)