import json
import random

import pandas as pd

SIZE = 430

dataset = pd.read_csv(
    "~/Box/dsi-core/11th-hour/good-food-purchasing/Filtered_OFF_with_sentences.csv",
    sep="\t",
    usecols=["nova_group", "sentence"],
)

random.seed(12345)

dataset1 = list(dataset[dataset["nova_group"] == 1].itertuples())
random.shuffle(dataset1)

dataset2 = list(dataset[dataset["nova_group"] == 2].itertuples())
random.shuffle(dataset2)

dataset3 = list(dataset[dataset["nova_group"] == 3].itertuples())
random.shuffle(dataset3)

dataset4 = list(dataset[dataset["nova_group"] == 4].itertuples())
random.shuffle(dataset4)

dataset = list(dataset1[0 : 3*SIZE] + dataset2[0 : 3*SIZE] + dataset3[0 : 3*SIZE] + dataset4[0 : 3*SIZE])
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
    for row in dataset[: 11*SIZE]:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.sentence),
                nova_group=int(row.nova_group),
            )
        )

with open("validation.jsonl", "w") as file:
    for row in dataset[11*SIZE : int(11.5*SIZE)]:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.sentence),
                nova_group=int(row.nova_group),
            )
        )

with open("test.jsonl", "w") as file:
    for row in dataset[int(11.5*SIZE) :]:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.sentence),
                nova_group=int(row.nova_group),
            )
        )
