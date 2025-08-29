"""
Gets an ingredient list for a branded food from the USDA.

The brand and product name are fuzzy-matched by a text-embedding space.
"""

import json
import os
import re
import sqlite3

import requests
import numpy as np
from sentence_transformers import SentenceTransformer

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

vector_database = np.lib.format.open_memmap(
    os.path.expanduser(
        "~/Box/dsi-core/11th-hour/good-food-purchasing/branded_food-all-MiniLM-L6-v2.npy"
    ),
    mode="r",
)

sqlite_database = sqlite3.connect(
    os.path.expanduser(
        "~/Box/dsi-core/11th-hour/good-food-purchasing/branded_food-all-MiniLM-L6-v2.sqlite"
    )
)
cursor = sqlite_database.cursor()


def usda_matches(
    vendor,
    brand,
    product,
    embedding_threshold=0.6,
    chatgpt_model="gpt-4.1-mini",
    chatgpt_temperature=1.0,
    chatgpt_num_trials=10,
    return_num_tokens=False,
):
    """
    For each `vendor`, `brand`, `product` triple, this function
        (1) embeds the text as a vector
        (2) searches for embedded USDA vectors with cosine similarity > `embedding_threshold`
        (3) sends all deduplicated text to ChatGPT to make a context-aware selection
        (4) returns a list of all matches, maybe empty

    This use of a vector database for large-scale search followed by an LLM is similar to RAG.

    ChatGPT is asked to provide `chatgpt_num_trials` responses. The `chatgpt_score` for a given `vendor`, `brand`, `product` triple is the fraction of times ChatGPT responded with that particular triple. The `chatgpt_score` can exceed 1.0 if a triple appears multiple times in the USDA dataset (possibly with different GTIN/UPC and/or ingredients).

    If you set `return_num_tokens` to True, this function returns a 2-tuple:
        * the normal list of output
        * the number of prompt tokens and completion tokens

    For gpt-4.1-mini on 2025-08-29, 1 million prompt tokens costs $0.40 and 1 million completion tokens costs $1.60.
    """

    vector = model.encode(f"{vendor} {brand} {product}")

    similarities = np.dot(vector_database, vector)

    indexes = np.nonzero(similarities >= embedding_threshold)[0]

    cursor.execute(
        f"SELECT vendor, brand, product FROM branded_food WHERE npy_index IN ({', '.join(['?'] * len(indexes))})",
        indexes.tolist(),
    )
    with_duplicates = ["{} {} {}".format(*row).strip() for row in cursor.fetchall()]
    unique_rows, inverse_indexes = np.unique_inverse(with_duplicates)

    choices = "\n    ".join(f"{i + 1}. {row}" for i, row in enumerate(unique_rows))
    message = f"""Which is the best match to the following food product description?

    {vendor} {brand} {product}

Here are your choices:

    {choices}

Respond with JSON indicating the number `N` corresponding to the best match or `null` if none of the choices are a good match."""

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        json={
            "model": chatgpt_model,
            "messages": [
                {"role": "user", "content": message},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "best_match",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "best": {"type": ["integer", "null"]},
                        },
                        "required": ["best"],
                        "additionalProperties": False,
                    },
                },
            },
            "temperature": chatgpt_temperature,
            "n": chatgpt_num_trials,
        },
    )
    response_json = response.json()

    best_indexes = [
        x
        for x in [
            json.loads(result["message"]["content"])["best"]
            for result in response_json["choices"]
        ]
        if x is not None
    ]
    if len(best_indexes) == 0:
        return []
    unique_indexes, index_counts = np.unique_counts(best_indexes)

    indexes_to_get = []
    scores = []
    for i, cnt in zip(unique_indexes, index_counts):
        for j in indexes[inverse_indexes == i - 1]:
            indexes_to_get.append(int(j))
            scores.append(float(cnt / chatgpt_num_trials))

    cursor.execute(
        f"SELECT * FROM branded_food WHERE npy_index IN ({', '.join(['?'] * len(indexes_to_get))})",
        indexes_to_get,
    )
    seen = set()
    out = []
    for row, score in zip(cursor.fetchall(), scores):
        if row in seen:
            continue
        seen.add(row[1:])
        out.append(
            {
                "encoding_similarity": similarities[row[0]],
                "chatgpt_score": score,
                "usda_index": row[0],
                "gtin_upc": row[1],
                "vendor": row[2],
                "brand": row[3],
                "product": row[4],
                "ingredients": row[5],
            }
        )

    if return_num_tokens:
        return out, {
            "prompt_tokens": response_json["usage"]["prompt_tokens"],
            "completion_tokens": response_json["usage"]["completion_tokens"],
        }
    else:
        return out


if __name__ == "__main__":
    import pandas as pd
    import csv
    import sys
    from tqdm import tqdm

    start, stop = map(int, sys.argv[1:])

    cgfp = pd.read_csv(
        "~/Box/dsi-core/11th-hour/good-food-purchasing/CONFIDENTIAL_GFPP Product Attribute List_8.26.25.csv",
        dtype=str,
    ).iloc[start:stop]

    with open(f"cgfp_usda_text_matches_{start}_{stop}.csv", "w") as file:
        writer = csv.writer(file)

        writer.writerow((
            "cgfp_index",
            "Product GTIN or UPC",
            "Vendor",
            "Brand Name",
            "Product Type",
            "Level of Processing",
            "encoding_similarity",
            "chatgpt_score",
            "usda_index",
            "usda_gtin_upc",
            "usda_vendor",
            "usda_brand",
            "usda_product",
            "usda_ingredients",
        ))

        for index, row in tqdm(cgfp.iterrows(), total=len(cgfp)):
            gtin_upc = "" if not isinstance(row["Product GTIN or UPC"], str) else row["Product GTIN or UPC"]
            cgfp_vendor = "" if not isinstance(row["Vendor"], str) else row["Vendor"]
            cgfp_brand = "" if not isinstance(row["Brand Name"], str) else row["Brand Name"]
            cgfp_product = "" if not isinstance(row["Product Type"], str) else row["Product Type"]
            cgfp_nova = "" if not isinstance(row["Level of Processing"], str) else row["Level of Processing"]

            try:
                matches = usda_matches(
                    cgfp_vendor,
                    cgfp_brand,
                    cgfp_product,
                    embedding_threshold=0.6,
                    chatgpt_model="gpt-4.1-mini",
                    chatgpt_temperature=1.0,
                    chatgpt_num_trials=10,
                    return_num_tokens=False,
                )
            except Exception as err:
                with open(f"failures/{index}", "w") as errfile:
                    errfile.write(f"{type(err).__name__}: {str(err)}")
                continue

            for match in matches:
                writer.writerow(
                    (
                        index,
                        gtin_upc,
                        cgfp_vendor,
                        cgfp_brand,
                        cgfp_product,
                        cgfp_nova,
                        match["encoding_similarity"],
                        match["chatgpt_score"],
                        match["usda_index"],
                        match["gtin_upc"],
                        match["vendor"],
                        match["brand"],
                        match["product"],
                        match["ingredients"],
                    )
                )
            file.flush()
