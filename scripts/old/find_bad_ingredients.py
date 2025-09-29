import os
import json
import csv

import requests
import pandas as pd
from tqdm import tqdm

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4.1-nano"

df = pd.read_csv("~/Box/dsi-core/11th-hour/good-food-purchasing/usfoods-ingredients.csv", dtype=str)

output_file = open("usfoods-ingredients-unwanted.csv", "w")
writer = csv.writer(output_file)
writer.writerow([
    "cgfp_index",
    "vendor_item_number",
    "ingredients",
    "food_dye",
    "caramel_color",
    "titanium_dioxide",
    "artificial_or_unspecified_flavors",
    "artificial_preservatives",
    "artificial_sweeteners",
    "emulsifiers",
    "flour_treatment_agents",
    "added_sodium",
    "added_sugars",
    "sugary_syrups",
    "caffeine",
    "natural_flavors",
    "phosphoric_acid_phosphates",
    "nitrites_nitrates_processed_meat",
    "refined_flour",
    "non_traditional_sugars",
    "thickening_agents",
    "natural_colorings",
    "hydrolyzed_vegetable_protein",
    "monosodium_glutamate",
    "mycoprotein",
    "preservatives",
])
output_file.flush()

for _, row in tqdm(df.iterrows(), total=len(df)):
    ingredients = row["ingredients"]
    if not isinstance(ingredients, str) or ingredients.strip() == "":
        continue

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
Given an ingredient list, indicate whether it contains one of the following unwanted ingredients as a JSON object.

* `"food_dye": true` for synthetic food dyes, such as blue 1, blue 2, green 3, red 3, red 40, yellow 5, yellow 6
* `"caramel_color": true` for caramel color CL 3-4
* `"titanium_dioxide": true` for titanium dioxide
* `"artificial_or_unspecified_flavors": true` for artificial and unspecified natural flavors
* `"artificial_preservatives": true` for artificial preservatives, such as butylated hydroxyanisole (BHA), butylated hydroxytoluene (BHT), propyl gallate, tert-butylhydroquinone (TBHQ)
* `"artificial_sweeteners": true` for artificial sweeteners and other sugar-free (non-nutritive, low-calorie and reduced-calorie) sweeteners: acesulfame potassium, advantame, aspartame, monk fruit extract, neotame, saccharin, stevia leaf extract (rebiana), sucralose, various sugar alcohols (erythritol, hydrogenated starch hydrolysate, isomalt, lactitol, maltitol, mannitol, sorbitol, xylitol) and thaumatin
* `"emulsifiers": true` for emulsifiers, such as brominated vegetable oil (BVO), carboxymethylcellulose (CMC) and polysorbates
* `"flour_treatment_agents": true` for flour treatment agents, such as bromated flour/potassium bromate, azodicarbonamide (ADA), potassium iodate
* `"added_sodium": true` for added sodium/salt
* `"added_sugars": true` for added sugars, such as agave, anhydrous dextrose, brown sugar, cane juice, cane sugar, confectioner's powdered sugar, corn syrup, corn syrup solids, crystal dextrose, date sugar, dextrose, evaporated cane juice, fructose, fruit juice concentrate, high-fructose corn syrup, high-maltose corn syrup, honey, invert sugar, isomaltulose, lactose, malt syrup, maltose, maple syrup, molasses, nectars (e.g. peach nectar, pear nectar), pancake syrup, raw sugar, sucrose, sugar, sugar cane juice, trehalose, and white granulated sugar
* `"sugary_syrups": true` for sugary syrups, such as high-fructose corn syrup, high-maltose corn syrup, high-dextrose corn syrup, corn syrup, tapioca syrup
* `"caffeine": true` for caffeine (especially added caffeine)
* `"natural_flavors": true` for specified natural flavors, such as essential oil, oleoresin, essence or extractive, protein hydrolysate, distillate, or any product of roasting, heating or enzymolysis, which contains the flavoring constituents derived from a spice, fruit or fruit juice, vegetable or vegetable juice, edible yeast, herb, bark, bud, root, leaf or similar plant material, meat, seafood, poultry, eggs, dairy products, or fermentation products thereof, whose significant function in food is flavoring rather than nutritional
* `"phosphoric_acid_phosphates": true` for phosphoric acid and phosphates
* `"nitrites_nitrates_processed_meat": true` for nitrites/nitrates and processed meat: meat that has been transformed through salting, curing, fermentation, smoking, or other processes to enhance flavor or improve preservation
* `"refined_flour": true` for refined or white (including bleached) flour: flour that has been treated with an oxidizing agent, most commonly benzoyl peroxide, but azodicarbonamide, chlorine dioxide or other agents
* `"non_traditional_sugars": true` for sugars metabolized differently from traditional sugars: allulose and tagatose
* `"thickening_agents": true` for thickening agents, such as carrageenan
* `"natural_colorings": true` for colorings (naturally derived), such as annatto, cochineal extract/carmine
* `"hydrolyzed_vegetable_protein": true` for hydrolyzed vegetable protein (HVP), often soybeans (not to be confused with isolated vegetable protein (IVP) or textured vegetable protein (TVP), both of which are safe)
* `"monosodium_glutamate": true` for monosodium glutamate (MSG)
* `"mycoprotein": true` for mycoprotein
* `"preservatives": true` for preservatives, such as benzoates (sodium benzoate, potassium benzoate, calcium benzoate) and benzoic acid, sulfites
""".strip()},
                    {"role": "user", "content": ingredients},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bad_ingredients",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "food_dye": {"type": "boolean"},
                                "caramel_color": {"type": "boolean"},
                                "titanium_dioxide": {"type": "boolean"},
                                "artificial_or_unspecified_flavors": {"type": "boolean"},
                                "artificial_preservatives": {"type": "boolean"},
                                "artificial_sweeteners": {"type": "boolean"},
                                "emulsifiers": {"type": "boolean"},
                                "flour_treatment_agents": {"type": "boolean"},
                                "added_sodium": {"type": "boolean"},
                                "added_sugars": {"type": "boolean"},
                                "sugary_syrups": {"type": "boolean"},
                                "caffeine": {"type": "boolean"},
                                "natural_flavors": {"type": "boolean"},
                                "phosphoric_acid_phosphates": {"type": "boolean"},
                                "nitrites_nitrates_processed_meat": {"type": "boolean"},
                                "refined_flour": {"type": "boolean"},
                                "non_traditional_sugars": {"type": "boolean"},
                                "thickening_agents": {"type": "boolean"},
                                "natural_colorings": {"type": "boolean"},
                                "hydrolyzed_vegetable_protein": {"type": "boolean"},
                                "monosodium_glutamate": {"type": "boolean"},
                                "mycoprotein": {"type": "boolean"},
                                "preservatives": {"type": "boolean"},
                            },
                            "required": [
                                "food_dye",
                                "caramel_color",
                                "titanium_dioxide",
                                "artificial_or_unspecified_flavors",
                                "artificial_preservatives",
                                "artificial_sweeteners",
                                "emulsifiers",
                                "flour_treatment_agents",
                                "added_sodium",
                                "added_sugars",
                                "sugary_syrups",
                                "caffeine",
                                "natural_flavors",
                                "phosphoric_acid_phosphates",
                                "nitrites_nitrates_processed_meat",
                                "refined_flour",
                                "non_traditional_sugars",
                                "thickening_agents",
                                "natural_colorings",
                                "hydrolyzed_vegetable_protein",
                                "monosodium_glutamate",
                                "mycoprotein",
                                "preservatives",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )
    except Exception as err:
        print(f"A {type(err).__name__}: {str(err)}")
        continue

    try:
        content = json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as err:
        print(f"B {type(err).__name__}: {str(err)}")
        continue

    try:
        food_dye = content["food_dye"]
        caramel_color = content["caramel_color"]
        titanium_dioxide = content["titanium_dioxide"]
        artificial_or_unspecified_flavors = content["artificial_or_unspecified_flavors"]
        artificial_preservatives = content["artificial_preservatives"]
        artificial_sweeteners = content["artificial_sweeteners"]
        emulsifiers = content["emulsifiers"]
        flour_treatment_agents = content["flour_treatment_agents"]
        added_sodium = content["added_sodium"]
        added_sugars = content["added_sugars"]
        sugary_syrups = content["sugary_syrups"]
        caffeine = content["caffeine"]
        natural_flavors = content["natural_flavors"]
        phosphoric_acid_phosphates = content["phosphoric_acid_phosphates"]
        nitrites_nitrates_processed_meat = content["nitrites_nitrates_processed_meat"]
        refined_flour = content["refined_flour"]
        non_traditional_sugars = content["non_traditional_sugars"]
        thickening_agents = content["thickening_agents"]
        natural_colorings = content["natural_colorings"]
        hydrolyzed_vegetable_protein = content["hydrolyzed_vegetable_protein"]
        monosodium_glutamate = content["monosodium_glutamate"]
        mycoprotein = content["mycoprotein"]
        preservatives = content["preservatives"]
    except Exception as err:
        print(f"C {type(err).__name__}: {str(err)}")
        continue

    writer.writerow([
        row["cgfp_index"],
        row["vendor_item_number"],
        row["ingredients"],
        food_dye,
        caramel_color,
        titanium_dioxide,
        artificial_or_unspecified_flavors,
        artificial_preservatives,
        artificial_sweeteners,
        emulsifiers,
        flour_treatment_agents,
        added_sodium,
        added_sugars,
        sugary_syrups,
        caffeine,
        natural_flavors,
        phosphoric_acid_phosphates,
        nitrites_nitrates_processed_meat,
        refined_flour,
        non_traditional_sugars,
        thickening_agents,
        natural_colorings,
        hydrolyzed_vegetable_protein,
        monosodium_glutamate,
        mycoprotein,
        preservatives,
    ])
    output_file.flush()

output_file.close()

print("DONE!")
