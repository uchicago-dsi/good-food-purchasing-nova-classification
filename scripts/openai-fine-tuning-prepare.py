import json
import random

import pandas as pd

dataset = pd.read_csv(
    "~/Box/dsi-core/11th-hour/good-food-purchasing/Filtered_OFF_with_sentences.csv",
    sep="\t",
    usecols=["nova_group", "ingredients_text"],
)

random.seed(12345)
dataset = list(dataset.itertuples())
random.shuffle(dataset)

# Encodings of desired outputs differ in only one token, the group integer.
# Thus, we can use the "logprobs" of that one token to quantify probability.
#
# encoding.encode('{"nova_group":1}') --> [10848, 86923, 15990, 1243, 16, 92]
# encoding.encode('{"nova_group":2}') --> [10848, 86923, 15990, 1243, 17, 92]
# encoding.encode('{"nova_group":3}') --> [10848, 86923, 15990, 1243, 18, 92]
# encoding.encode('{"nova_group":4}') --> [10848, 86923, 15990, 1243, 19, 92]

message_format = '{{"messages":[{{"role":"user","content":{ingredients}}},{{"role":"assistant","content":"{{\\\"nova_group\\\":{nova_group}}}"}}]}}\n'

with open("training.jsonl", "w") as file:
    for row in dataset[0:1000]:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.ingredients_text),
                nova_group=int(row.nova_group),
            )
        )

with open("validation.jsonl", "w") as file:
    for row in dataset[1000:2000]:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.ingredients_text),
                nova_group=int(row.nova_group),
            )
        )

with open("test.jsonl", "w") as file:
    for row in dataset[2000:3000]:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.ingredients_text),
                nova_group=int(row.nova_group),
            )
        )
