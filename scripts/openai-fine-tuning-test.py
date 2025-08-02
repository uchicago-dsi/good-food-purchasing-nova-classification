import os
import json
import threading
import queue

import numpy as np
import requests

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4.1-nano"
MODEL_DIR = "gpt-4.1-nano"
TRY = "try5"
NUM_THREADS = 100


def probability_distribution(ingredients, countdown):
    # * `{"nova_group":1}` for unprocessed or minimally processed foods, containing only raw or crushed, chilled, frozen, or dried vegetables, meat, seafood, milk, seeds, or spices, etc., without added sweeteners or flavors.
    # * `{"nova_group":2}` for processed culinary ingredients, such as vegetable oils, butter, lard, sugar, molasses, honey, or syrups, which can include anti-oxidants, salt, and added vitamins or minerals.
    # * `{"nova_group":3}` for processed foods such as canned or bottled vegetables and legumes in brine, salted or sugared nuts and seeds, salted, dried cured, or smoked meats and fish, canned fish (with or without preservatives), fruit in syrup (with or without added anti-oxidants), and freshly made unpackaged breads and cheeses.
    # * `{"nova_group":4}` for ultra-processed foods, often ready-to-consume products like carbonated soft drinks, sweet or savory packaged snacks, candies, ice cream, mass-produced breads, margarines and other spreads, cookies, pastries, breakfast cereals, energy bars, energy drinks, instant sauces, ready-to-heat pasta and pizzas, pultry and fish "nuggets" or "sticks", sausages, burgers, hot dogs, infant formulas, health and "slimming" products such as meal-replacement shakes and powders.

    if countdown == 0:
        raise Exception(f"failed on: {ingredients}")
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": """
Your job is to identify a food product's NOVA classification, given its ingredient lists, as one of the four following JSON objects (with no whitespace):
* `{"nova_group":1}` for unprocessed or minimally processed foods
* `{"nova_group":2}` for processed culinary ingredients
* `{"nova_group":3}` for processed foods
* `{"nova_group":4}` for ultra-processed foods
""".strip()},
                    {"role": "user", "content": ingredients},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "nova_classification",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "nova_group": {
                                    "type": "integer",
                                    "enum": [1, 2, 3, 4],
                                },
                            },
                            "required": ["nova_group"],
                            "additionalProperties": False,
                        },
                    },
                },
                "logprobs": True,
                "top_logprobs": 10,
            },
        )
    except Exception:
        return probability_distribution(ingredients, countdown - 1)

    if response.status_code != 200:
        return probability_distribution(ingredients, countdown - 1)

    data = response.json()
    if len(data.get("choices", [])) != 1:
        return probability_distribution(ingredients, countdown - 1)

    for token in data["choices"][0]["logprobs"]["content"]:
        if token["token"] in ("1", "2", "3", "4"):
            return {
                int(x["token"]): np.exp(x["logprob"])
                for x in token["top_logprobs"]
                if x["token"] in ("1", "2", "3", "4")
            }

    return probability_distribution(ingredients, countdown - 1)


def worker(which, tasks):
    with open(f"test-results/{TRY}/{MODEL_DIR}/thread-more-{which}.csv", "w") as file:
        while True:
            task = tasks.get()
            if task is None:
                break
            index, ingredients, truth = task
            distribution = probability_distribution(ingredients, 5)
            for value in (1, 2, 3, 4):
                if value not in distribution:
                    distribution[value] = 0
            file.write(
                f"{index},{distribution[1]},{distribution[2]},{distribution[3]},{distribution[4]},{truth}\n"
            )
            file.flush()


tasks = queue.Queue()
with open(f"{TRY}/test-more.jsonl") as file:
    index = 0
    for line in file:
        user_message, assistant_message = json.loads(line)["messages"]
        tasks.put(
            (index, user_message["content"], json.loads(assistant_message["content"])["nova_group"])
        )
        index += 1

for _ in range(NUM_THREADS):
    tasks.put(None)

threads = []
for which in range(NUM_THREADS):
    threads.append(threading.Thread(target=worker, args=(which, tasks)))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
