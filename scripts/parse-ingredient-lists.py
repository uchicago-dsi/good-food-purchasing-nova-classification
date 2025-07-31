import re
import os
import json
import random

import requests
import pandas as pd

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

branded_food = pd.read_csv(
    "~/Box/dsi-core/11th-hour/good-food-purchasing/branded_food.csv",
    usecols=["fdc_id", "ingredients"],
)

branded_food_list = list(branded_food[branded_food["ingredients"].notna()].itertuples())
random.shuffle(branded_food_list)

# save memory; only need one copy
del branded_food

for row in branded_food_list:
    outfilename = f"ingredient-lists/{row.fdc_id}.json"
    if os.path.exists(outfilename):
        continue

    # When running many instances of this script, writing this empty file is a
    # synchronization point because of POSIX. Each will only be handled once.
    with open(outfilename, "w") as file:
        pass

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            json={
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": """
The user will provide a list of food ingredients from a Nutrition Facts label.

Your job is to parse it into a JSON list of individual ingredients for further processing.

Some ingredients are followed by parenthesized (or bracketed) text consisting of nested ingredient lists. Parse the sub-ingredients in these lists and put everything into a single, flattened, list of strings. For instance, if given

```
MILK, DRIED CAYENNE PEPPER SAUCE (RED PEPPERS, VINEGAR, SALT, DRIED GARLIC)
```

you should produce

```json
["MILK", "DRIED CAYENNE PEPPER SAUCE", "RED PEPPERS", "VINEGAR", "SALT", "DRIED GARLIC"]
```

The text also contains words that are not part of any ingredient, such as `ALL-NATURAL GLUTEN-FREE INGREDIENTS: ` or `CONTAINS LESS THAN 2% OF ` or ` AND/OR `, which should not be included in the output. Also drop words that explain why an ingredient was added, such as ` ADDED AS A PRESERVATIVE` or ` TO PRESERVE FRESHNESS`. Also drop words specifying how much of the ingredient is in the food, such as "99%". These words should be dropped even if they appear as parenthesized (or bracketed) text.

The strings in the output list should be exact substrings of the input text, in the same order.
""".strip()},
                    {"role": "user", "content": row.ingredients},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "list_of_ingredients",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "ingredients": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        )
    except:
        print(f"{row.fdc_id}: connection failed", flush=True)
        continue

    if response.status_code != 200:
        print(f"{row.fdc_id}: status code is {response.status_code}", flush=True)
        continue

    data = response.json()

    if len(data.get("choices", [])) == 0:
        print(f"{row.fdc_id}: response has no 'choices'", flush=True)
        continue

    content = json.loads(data["choices"][0].get("message", {}).get("content", "{}"))

    if not isinstance(content.get("ingredients"), list):
        print(f"{row.fdc_id}: choices has no 'ingredients'", flush=True)
        continue

    if not all(isinstance(x, str) for x in content["ingredients"]):
        print(f"{row.fdc_id}: not a flat list (didn't follow JSON schema)")
        continue

    m = re.search("(.*)".join(re.escape(x) for x in content["ingredients"]), row.ingredients, re.I)

    if m is None:
        print(f"{row.fdc_id}: {content['ingredients']} <<<<< {json.dumps(row.ingredients)}", flush=True)
        continue

    content["separators"] = list(m.groups())

    with open(outfilename, "w") as file:
        json.dump(content, file)
