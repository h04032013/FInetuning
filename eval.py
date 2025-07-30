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

#Check is_equiv with hendrycks script, grade with gpt api

#helper functions
def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def file_contains_string(file_path, id):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()
            return id in contents
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False

#Just compare accuracy only using is_equiv, Phi standard is 64%
def eval_response(input_path, output_path, batch_size, mistake_path):

    def chunked(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    problem_data = load_json(input_path)

    response_data = load_json(output_path)

    problem_lookup = {item["unique_id"]: item["answer"] for item in problem_data}
    correct_total = 0
    wrong_answers =[]
    accuracy = 0.0

    for batch in tqdm(chunked(response_data, batch_size), total=len(response_data)//batch_size,desc="Evaluating"):
        for item in batch:
            unique_id = item["unique_id"]
            final_answer = item.get("answer")
            correct_answer = problem_lookup.get(unique_id)

            if is_equiv(final_answer, correct_answer):
                correct_total = correct_total+ 1
                accuracy = (correct_total / len(response_data)) * 100
            else: #Print for smaller debugging & prototyping
                #print("Wrong: ", unique_id)
                #print("Guess: " , final_answer, "Correct: ", correct_answer)

                wrong_answers.append({
                    "problem": item["problem"],
                    "level": item["level"],
                    "type": item["type"],
                    "solution": item["solution"],
                    "answer": item["answer"],
                    "unique_id": item["unique_id"],
                    "correct": correct_answer
                })

    save_json(wrong_answers, mistake_path)

    print(f"\nTotal correct: {correct_total} / {len(response_data)}")
    print(f"Accuracy: {accuracy:.2f}%")

def build_prompt(problem: str, response: str, rubric_template: str) -> str:
    return rubric_template.format(problem=problem, response=response)

rubric_template = """
You are a math teacher grading a student's solution.

Grade the response using this rubric (0–5 for each):

    Correctness: Is the final answer correct? Are the steps mathematically valid?
    Clarity: Is the explanation easy to follow and well-structured?
    Mathematical Reasoning: Are the steps justified and logically sound?
    Notation: Is mathematical notation used correctly, consistently, and appropriately?
    Total: Sum of previous scores

Respond in this format:

Correctness: X
Clarity: Y
Reasoning: Z
Notation: W
Total: S
Comment: [Very brief explanation of the response's strengths and weaknesses]

Here is the problem:
{problem}

Here is the student's response:
{response}
"""

MODEL = "gpt-4.1-mini-2025-04-14"  # or "gpt-4"
RATE_LIMIT = 25  # max requests per minute (adjust based on your OpenAI tier)
MAX_CONCURRENT_REQUESTS = 3  # how many requests to send in parallel
RETRY_ATTEMPTS = 5
SAVE_EVERY = 50  # how often to save progress

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_fixed(8))
async def call_openai(api_path, session, semaphore, prompt, retries=RETRY_ATTEMPTS):

    with open (api_path, 'r') as f:
        API_KEY = f.read()


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

#extract number scores to save in json
def extract_scores(str_grade):
    scores =[]
    str_grade = str_grade.split()
    Correctness = str_grade[1]
    Clarity = str_grade[3]
    Reasoning = str_grade[5]
    Notation = str_grade[7]
    Total = str_grade[9]

    scores = [Correctness, Clarity, Reasoning, Notation, Total]
    return scores

#instrcutions for 1 entry
async def grade_entry(api_path, entry, session, semaphore, true_lookup, mistake_path):
    problem = entry.get("problem", "")
    response = entry.get("solution", "")
    prediction = entry.get("answer","")
    correct_answer = true_lookup.get(entry["unique_id"], "")

    prompt = rubric_template.format(problem=problem, response=response)
    grading = await call_openai(api_path, session, semaphore, prompt)
    #extraction method
    entry["grading"] = grading
    entry["correctness"], entry["clarity"], entry["reasoning"], entry["notation"], entry["total"] = extract_scores(grading)

    is_correct = is_equiv(prediction, correct_answer)
    entry["api_equiv"] = not file_contains_string(mistake_path, entry.get("unique_id",""))
    entry["is_equiv"] = is_correct

    if not is_correct:
        entry["true_answer"] = correct_answer

    return entry

#collective api calls
async def grade_all(api_path, data, true_lookup, graded_path, mistake_path):
    semaphore = asyncio.Semaphore(RATE_LIMIT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        for i, entry in enumerate(asyncio.as_completed([
            grade_entry(api_path, e, session, semaphore, true_lookup, mistake_path) for e in data
        ])):
            graded = await entry
            results.append(graded)
            if len(results) % SAVE_EVERY == 0:
                print(f"Saving checkpoint at {len(results)} entries...")
                save_json(results, graded_path)
        return results

#collect questions and run all api calls then save output scores
def score_final_call(api_path, output_path, graded_path, input_path, mistake_path):

    print("Loading input file...")
    
    data = load_json(output_path)
    true_data = load_json(input_path)

    true_lookup = {item["unique_id"]: item["answer"] for item in true_data}

    results = asyncio.run(grade_all(api_path, data, true_lookup, graded_path, mistake_path))

    save_json(results, graded_path)