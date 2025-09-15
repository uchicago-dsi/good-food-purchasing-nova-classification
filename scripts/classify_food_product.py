import csv
import io
import json
import os
import re
import sys
import time

import jwt
import requests
import pandas as pd

INSTALLATION_ID = "86079416"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
APP_PRIVATE_KEY = os.environ["APP_PRIVATE_KEY"]
APP_CLIENT_ID = os.environ["APP_CLIENT_ID"]
DISCUSSION_BODY = os.environ["DISCUSSION_BODY"]
DISCUSSION_ID = os.environ["DISCUSSION_ID"]

NEW_DISCUSSION_URL = "https://github.com/uchicago-dsi/good-food-purchasing-nova-classification/discussions/new?category=classify-food-product"
MODEL = "ft:gpt-4.1-nano-2025-04-14:u-chicago:cgfp-name-to-nova-try2:CEzMM3Qh"
NUM_CHATGPT_RETRIES = 5

jwt_instance = jwt.JWT()
current_token = None
expiration = 0


def get_token():
    global current_token
    global expiration
    now = int(time.time())
    if now >= expiration - 1:
        expiration = now + 600
        current_jwt = jwt_instance.encode(
            {"iat": now, "exp": expiration, "iss": APP_CLIENT_ID},
            jwt.jwk_from_pem(APP_PRIVATE_KEY.encode()),
            alg="RS256",
        )
        response = requests.post(
            f"https://api.github.com/app/installations/{INSTALLATION_ID}/access_tokens",
            headers={
                "Authorization": f"Bearer {current_jwt}",
                "Accept": "application/vnd.github+json",
            },
        )
        current_token = response.json().get("token")
    return current_token


def write_comment(text):
    response = requests.post(
        "https://api.github.com/graphql",
        headers={
            "Authorization": f"token {get_token()}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "query": """
mutation AddDiscussionComment {
  addDiscussionComment(input: {
    discussionId: "%s",
    body: "%s"
  }) {
    comment { id }
  }
}
"""
            % (DISCUSSION_ID, text)
        },
    )
    print(f"write_comment {response.status_code = } {response.text = }")


class ChatGPTError(Exception):
    def __init__(self, index):
        self.index = index


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


class Tee(io.StringIO):
    # def __init__(self, filename):
    #     self.file = open(filename, "w")

    # def close(self):
    #     self.file.close()

    def __enter__(self):
        pass
        # return self.file.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        pass
        # return self.file.__exit__(self, exc_type, exc_value, traceback)

    def write(self, data):
        super().write(data)
        print(data, end="")


if __name__ == "__main__":
    m = re.search(
        r"\[[^]]*\]\((https://github.com/user-attachments/[^)]+\.csv)\)",
        DISCUSSION_BODY,
    )
    if m is None:
        write_comment(
            f"You need to attach a CSV file in your message. [Create a new discussion]({NEW_DISCUSSION_URL}) and drag a CSV file into it or click on the button below the message to choose a file."
        )
        sys.exit()

    attachment_url = m.group(1)
    print(f"Getting CSV data from {attachment_url}")

    try:
        response = requests.get(
            attachment_url,
            headers={
                "Authorization": f"token {get_token()}",
                "Accept": "application/vnd.github+json",
            },
        )
    except Exception as err:
        write_comment(
            f"Attempted to get [{attachment_url}]({attachment_url}), but it failed to fetch with {type(err).__name__}: {str(err)}\n\nIf you know how to fix this error, do so [in a new discussion]({NEW_DISCUSSION_URL})."
        )
        sys.exit()

    print(f"CSV content is {response.content}")

    try:
        df = pd.read_csv(io.BytesIO(response.content), dtype=str)
    except Exception as err:
        write_comment(
            f"Attempted to read [{attachment_url}]({attachment_url}), but Pandas failed to read it with {type(err).__name__}: {str(err)}\n\nIf you know how to fix this error, do so [in a new discussion]({NEW_DISCUSSION_URL})."
        )
        sys.exit()

    print(f"Columns in Pandas are {df.columns}")

    needs = []
    for column in ("Processor", "Brand Name", "Product Type"):
        if column not in df.columns:
            needs.append(column)
    if len(needs) != 0:
        write_comment(
            f"The following columns are missing: {', '.join(f'`{x}`' for x in needs)} (case-sensitive); found the following columns: {', '.join(f'`{x}`' for x in df.columns)}. Fix the CSV file and upload it [in a new discussion]({NEW_DISCUSSION_URL})."
        )
        sys.exit()

    df["message"] = (
        df["Processor"].fillna("").str[:]
        + "\n"
        + df["Brand Name"].fillna("").str[:]
        + "\n"
        + df["Product Type"].fillna("").str[:]
    )

    with Tee() as file:
        out = csv.writer(file)
        out.writerow(
            ["index", "Processor", "Brand Name", "Product Type", "nova_from_chatgpt"]
        )

        failures = []
        for index, row in df.iterrows():
            try:
                result = chatgpt_response(index, row("message"), NUM_CHATGPT_RETRIES)
            except ChatGPTError as err:
                failures.append(err.index)
            except Exception as err:
                write_comment(
                    f"During processing, we encountered {type(err).__name__}: {str(err)}\n\nIf you know how to fix this error, do so [in a new discussion]({NEW_DISCUSSION_URL})."
                )
                sys.exit()

            out.writerow(
                [
                    index,
                    row("Processor"),
                    row("Brand Name"),
                    row("Product Type"),
                    result,
                ]
            )

        if len(failures) != 0:
            preamble = f"The following indexes (first row is zero) failed to be processed (likely a timeout when connecting to ChatGPT): {', '.join(map(str, failures))}\n\n"
        else:
            preamble = ""

        write_comment(
            f"""{preamble}Here's your data with NOVA scores from ChatGPT:

```csv
{file.getvalue()}```
"""
        )
