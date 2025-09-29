import csv
import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import requests

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "ft:gpt-4.1-nano-2025-04-14:u-chicago:cgfp-name-to-nova-try2:CEzMM3Qh"
NUM_THREADS = 30
NUM_CHATGPT_RETRIES = 5

NOVA_NAMES = {
    1: "Whole/Minimally Processed",
    2: "Culinary Ingredient",
    3: "Moderately Processed",
    4: "Ultra-Processed",
}


def chatgpt_response(index, message, countdown):
    if countdown == 0:
        raise ChatGPTError(index)
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
                    {
                        "role": "system",
                        "content": """
Your job is to identify a food product's NOVA classification, given its vendor, brand name, and description, as one of the four following JSON objects (with no whitespace):
* `{"nova_group":1}` for unprocessed or minimally processed foods, containing only raw or crushed, chilled, frozen, or dried vegetables, meat, seafood, milk, seeds, or spices, etc., without added sweeteners or flavors.
* `{"nova_group":2}` for processed culinary ingredients, such as vegetable oils, butter, lard, sugar, molasses, honey, or syrups, which can include anti-oxidants, salt, and added vitamins or minerals.
* `{"nova_group":3}` for processed foods, such as canned or bottled vegetables and legumes in brine, salted or sugared nuts and seeds, salted, dried cured, or smoked meats and fish, canned fish (with or without preservatives), fruit in syrup (with or without added anti-oxidants), and freshly made unpackaged breads and cheeses.
* `{"nova_group":4}` for ultra-processed foods, often ready-to-consume products like carbonated soft drinks, sweet or savory packaged snacks, candies, ice cream, mass-produced breads, margarines and other spreads, cookies, pastries, breakfast cereals, energy bars, energy drinks, instant sauces, ready-to-heat pasta and pizzas, pultry and fish "nuggets" or "sticks", sausages, burgers, hot dogs, infant formulas, health and "slimming" products such as meal-replacement shakes and powders.
""".strip(),
                    },
                    {"role": "user", "content": message},
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
            },
        )
    except Exception:
        return chatgpt_response(index, message, countdown - 1)

    if response.status_code != 200:
        return chatgpt_response(index, message, countdown - 1)

    try:
        return json.loads(response.json()["choices"][0]["message"]["content"])[
            "nova_group"
        ]
    except Exception:
        return chatgpt_response(index, message, countdown - 1)


def worker(which, tasks, num_tasks):
    with open(f"results/thread-{which}.csv", "w") as file:
        writer = csv.writer(file)
        while True:
            task = tasks.get()
            if task is None:
                break

            index, row = task
            print(f"{time.strftime('%H:%M:%S')} {100 * index / num_tasks:.0f}% {index}")

            result = chatgpt_response(index, row["message"], NUM_CHATGPT_RETRIES)
            writer.writerow(
                (
                    index,
                    row["Processor"],
                    row["Brand Name"],
                    row["Product Type"],
                    NOVA_NAMES.get(result, result),
                )
            )
            file.flush()


df = pd.read_csv("~/Downloads/CCH_DSI Data Sample.csv", dtype=str)
df["message"] = (
    df["Processor"].fillna("").str[:]
    + "\n"
    + df["Brand Name"].fillna("").str[:]
    + "\n"
    + df["Product Type"].fillna("").str[:]
)

tasks = queue.Queue()
for index, row in df.iterrows():
    tasks.put((index, row))

for _ in range(NUM_THREADS):
    tasks.put(None)

threads = []
for which in range(NUM_THREADS):
    threads.append(threading.Thread(target=worker, args=(which, tasks, len(df))))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
