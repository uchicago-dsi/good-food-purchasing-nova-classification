import json

import numpy as np
import pandas as pd

df = pd.read_csv(
    "~/Box/dsi-core/11th-hour/good-food-purchasing/CONFIDENTIAL_GFPP Product Attribute List_8.26.25.csv",
    dtype=str,
)[
    [
        "Processor",
        "Brand Name",
        "Product Type",
        "Food Product Category",
        "Primary Food Product Category",
        "Level of Processing",
    ]
].dropna(
    subset="Level of Processing"
)
df["Level of Processing"] = df["Level of Processing"].map(
    {
        "Whole/Minimally Processed": 1,
        "Culinary Ingredient": 2,
        "Moderately Processed": 3,
        "Ultra-Processed": 4,
    }
)
df = df[df["Level of Processing"].notna()]
df["Level of Processing"] = df["Level of Processing"].astype(int)
df["message"] = (
    df["Processor"].fillna("").str[:]
    + "\n"
    + df["Brand Name"].fillna("").str[:]
    + "\n"
    + df["Product Type"].fillna("").str[:]
)
df = df.drop(columns=["Processor", "Brand Name", "Product Type"])

level1 = df[df["Level of Processing"] == 1]
level2 = df[df["Level of Processing"] == 2]
level3 = df[df["Level of Processing"] == 3]
level4 = df[df["Level of Processing"] == 4]

level1 = level1.take(np.random.permutation(len(level1)))
level2 = level2.take(np.random.permutation(len(level2)))
level3 = level3.take(np.random.permutation(len(level3)))
level4 = level4.take(np.random.permutation(len(level4)))

minimum_test_size = 100
validation_size = 100
training_size = (
    min(len(level1), len(level2), len(level3), len(level4))
    - minimum_test_size
    - validation_size
)

training = pd.concat(
    [
        level1.iloc[:training_size],
        level2.iloc[:training_size],
        level3.iloc[:training_size],
        level4.iloc[:training_size],
    ]
)
training = training.take(np.random.permutation(len(training)))

validation = pd.concat(
    [
        level1.iloc[training_size : training_size + validation_size],
        level2.iloc[training_size : training_size + validation_size],
        level3.iloc[training_size : training_size + validation_size],
        level4.iloc[training_size : training_size + validation_size],
    ]
)
validation = validation.take(np.random.permutation(len(validation)))

test = pd.concat(
    [
        level1.iloc[training_size + validation_size :],
        level2.iloc[training_size + validation_size :],
        level3.iloc[training_size + validation_size :],
        level4.iloc[training_size + validation_size :],
    ]
)
test = test.take(np.random.permutation(len(test)))

message_format = '{{"messages":[{{"role":"user","content":{description}}},{{"role":"assistant","content":"{{\\"nova_group\\":{nova_group}}}"}}]}}\n'

with open("cgfp-training.jsonl", "w") as file:
    for _, row in training.iterrows():
        file.write(
            message_format.format(
                description=json.dumps(row["message"]),
                nova_group=int(row["Level of Processing"]),
            )
        )

with open("cgfp-validation.jsonl", "w") as file:
    for _, row in validation.iterrows():
        file.write(
            message_format.format(
                description=json.dumps(row["message"]),
                nova_group=int(row["Level of Processing"]),
            )
        )

test.reset_index().to_csv("cgfp-test.csv", index=False)
