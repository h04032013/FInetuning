import json
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

#Check black/white accuracy with llm

API_KEY_PATH = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/openai_key"

with open(API_KEY_PATH, 'r') as f:
    API_KEY = f.read().strip()

client = AsyncOpenAI(api_key=API_KEY)

#Helper for grading later

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def api_equiv(prompt):
    response = await client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a teacher grading accuracy on answers to math questions."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip().lower()
    return content.startswith("true")

async def evaluate_response(input_path, output_path, batch_size, mistake_path, max_concurrent_requests=5):
    def chunked(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    with open(input_path, "r") as f:
        problem_data = json.load(f)

    with open(output_path, "r") as f:
        response_data = json.load(f)

    problem_lookup = {item["unique_id"]: item["answer"] for item in problem_data}
    correct_total = 0
    wrong_answers = []

    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def evaluate_item(item):
        nonlocal correct_total
        unique_id = item["unique_id"]
        final_answer = item.get("answer")
        correct_answer = problem_lookup.get(unique_id)
        problem = item["problem"]

        prompt = f"""Given the question: {problem}
        Compare the following two math answers. Only respond "TRUE" if both answers are mathematically equivalent, otherwise return "FALSE"
        Answer 1: {final_answer}
        Answer 2: {correct_answer}
        """

        async with semaphore:
            try:
                result = await api_equiv(prompt)
                if result:
                    correct_total += 1
                else:
                    wrong_answers.append({
                        "problem": item["problem"],
                        "level": item["level"],
                        "type": item["type"],
                        "solution": item["solution"],
                        "answer": item["answer"],
                        "unique_id": item["unique_id"],
                        "correct": correct_answer
                    })
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {unique_id}: {e}")
                # You may want to append to wrong_answers or log this elsewhere

    for batch in tqdm(chunked(response_data, batch_size), total=len(response_data) // batch_size, desc="Evaluating"):
        await asyncio.gather(*(evaluate_item(item) for item in batch))

    with open(mistake_path, "w") as f:
        json.dump(wrong_answers, f, indent=2)

    accuracy = (correct_total / len(response_data)) * 100
    print(f"\nTotal correct: {correct_total} / {len(response_data)}")
    print(f"Accuracy: {accuracy:.2f}%")
