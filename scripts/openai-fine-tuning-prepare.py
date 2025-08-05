import json
import random

import pandas as pd

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

test = dataset1[0:100] + dataset2[0:100] + dataset3[0:100] + dataset4[0:100]
random.shuffle(test)

validation = dataset1[100:200] + dataset2[100:200] + dataset3[100:200] + dataset4[100:200]
random.shuffle(validation)

training2 = dataset2[200:] * 11
training1 = dataset1[200 : 200 + len(training2)]
training3 = dataset3[200 : 200 + len(training2)]
training4 = dataset4[200 : 200 + len(training2)]
training = training1 + training2 + training3 + training4
random.shuffle(training)

# Encodings of desired outputs differ in only one token, the group integer.
# Thus, we can use the "logprobs" of that one token to quantify probability.
#
# encoding.encode('{"nova_group":1}') --> [10848, 86923, 15990, 1243, 16, 92]
# encoding.encode('{"nova_group":2}') --> [10848, 86923, 15990, 1243, 17, 92]
# encoding.encode('{"nova_group":3}') --> [10848, 86923, 15990, 1243, 18, 92]
# encoding.encode('{"nova_group":4}') --> [10848, 86923, 15990, 1243, 19, 92]

message_format = '{{"messages":[{{"role":"user","content":{ingredients}}},{{"role":"assistant","content":"{{\\\"nova_group\\\":{nova_group}}}"}}]}}\n'

with open("training.jsonl", "w") as file:
    for row in training:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.sentence),
                nova_group=int(row.nova_group),
            )
        )

with open("validation.jsonl", "w") as file:
    for row in validation:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.sentence),
                nova_group=int(row.nova_group),
            )
        )

with open("test.jsonl", "w") as file:
    for row in test:
        file.write(
            message_format.format(
                ingredients=json.dumps(row.sentence),
                nova_group=int(row.nova_group),
            )
        )
