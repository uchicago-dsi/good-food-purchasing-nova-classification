import time

import requests
import pandas as pd

cgfp = pd.read_csv(
    "~/Box/dsi-core/11th-hour/good-food-purchasing/CONFIDENTIAL_GFPP Product Attribute List_8.26.25.csv",
    dtype=str,
)[
    [
        # "Product GTIN or UPC",
        "Vendor",
        "Vendor Item Number",
        # "Brand Name",
        # "Product Type",
        # "Level of Processing",
    ]
]
# cgfp["Product GTIN or UPC"] = cgfp["Product GTIN or UPC"].apply(
#     lambda x: f"{int(float(x.replace(' ', ''))):014d}" if isinstance(x, str) and x != "#REF!" else pd.NA
# )

cgfp_usfoods = cgfp.query("Vendor == 'US Foods'")

for index, row in cgfp_usfoods.iterrows():
    item_number = row["Vendor Item Number"]

    try:
        response = requests.get(
            f"https://panamax-api.ama.usfoods.com/product-domain-api/v1/productdetail?productNumber={item_number}",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.eyJ1c2VyIjp7InVzZXJOYW1lIjoiR1VFU1RVU0VSIiwidXNlcklkIjoxMzA3NDc0MiwiaXBBZGRyZXNzIjpudWxsfSwic2NvcGVzIjpbInVzZi11c2VyIiwidXNmLWN1c3RvbWVyIiwidXNmLXByb2R1Y3QiXSwidXNmLWNsYWltcyI6eyJkaXZpc2lvbk51bWJlciI6MjI1MCwiY3VzdG9tZXJOdW1iZXIiOjYxNjgxNzA2LCJkZXBhcnRtZW50TnVtYmVyIjowLCJ1c2VyTmFtZSI6IkdVRVNUVVNFUiIsInVzZXJJZCI6MTMwNzQ3NDIsIm9yZGVyVGFrZXJSb2xlIjpudWxsLCJvcmRlclRha2VySWQiOm51bGwsIm9yZGVyVGFrZXJTb3VyY2UiOm51bGwsInVzZXJUeXBlIjoiZ3Vlc3QiLCJyZXF1aXJlQ3VzdG9tZXJQTyI6bnVsbCwiY2xpZW50SWQiOm51bGwsImNsaWVudENvbmNlcHQiOm51bGwsImhhc0ludmVudG9yeSI6ZmFsc2UsImN1c3RvbWVyVHlwZSI6IlNUIiwib2dQcmludFByaWNlSW5kIjoiTiIsInJlc3RyaWN0VG9PRyI6Ik4iLCJkaXJlY3RFbGlnaWJsZSI6dHJ1ZSwiZW1haWwiOiJ2aXZpYW5lLnJhZ3Vzb0B1c2Zvb2RzLmNvbSIsImFsbG93SW1wZXJzb25hdGlvbiI6ZmFsc2UsInRtVXNlciI6Ik4iLCJyNFJlZGlyZWN0IjoiWSIsImhpZGVJbnZlbnRvcnkiOiJZIiwic3Vic3RpdHV0aW9uQWxsb3dlZCI6IlkiLCJkaXNwbGF5RHdvU3RhdHVzIjpmYWxzZSwiaGlkZVN1cHBsaWVyVW5hdmFpbGFibGUiOmZhbHNlLCJoaWRlT3V0T2ZTdG9jayI6ZmFsc2UsInJlc3RyaWN0TUxNQ2F0YWxvZ1NlYXJjaCI6ZmFsc2UsImVjb21Vc2VyVHlwZSI6IjEiLCJpc1N1cGVyVXNlciI6ZmFsc2UsImlzR3Vlc3RVc2VyIjp0cnVlLCJtc2xSZXN0cmljdGlvbk92ZXJyaWRlIjpmYWxzZSwibXUiOiIiLCJpc1VTRm9vZHNBUiI6ZmFsc2UsImJhc2VsaW5lIjpmYWxzZSwic2VydmljZUFjY291bnQiOmZhbHNlLCJwdW5jaHRocnVTZXNzaW9uIjpmYWxzZSwibWVzc2FnaW5nQWRtaW4iOmZhbHNlLCJjdXN0b21lckFkbWluIjpmYWxzZSwicHVuY2hvdXRTZXNzaW9uIjpmYWxzZSwiaW50ZWdyYXRpb25TZXNzaW9uIjpmYWxzZSwidGVybXNPZlVzZUFjY2VwdGVkIjp0cnVlLCJwYXlJbnZvaWNlVXNlciI6ZmFsc2UsInRyZW5kdmlldzM2MFVzZXIiOmZhbHNlLCJsb2FkUmVjZW50bHlQdXJjaGFzZWRJbk9HIjoiWSJ9LCJpc3MiOiJodHRwczovL29yZGVyLnVzZm9vZHMuY29tIiwic3ViIjoiMTMwNzQ3NDIiLCJhdWQiOlsiaHR0cHM6Ly9vcmRlci51c2Zvb2RzLmNvbSJdLCJuYmYiOjE3NTY4NDU3NjcsImlhdCI6MTc1Njg0NTc2NywiZXhwIjoxNzU2OTMyMTY3fQ.mn-qmu5hx9xPMjs3J3wjjfq8kOhj2u6F9AAK945Y-1SZF_61lZ2fk1NGnLL6xq_GwNtJKUtz-YQWGH6BVV04q6uWYOgI9KKiGAaF-xelYM3SmFRomzJ1skNW3ebpxGdtQbGw_Cb9HVB6xJQ6dRBqHQA39Xd6GjuxhIwvcCcxl9c5AvPS4e7okKGiWexVQju-3nngoGWZNux5g_4dNSshIyEBtH1AaOP6CpcVorQ67tzXh6M1u9pTDgumVq4Sgh2mlC6UvHHz6QwSz58rYF9qNUdaRO6Uhttjtren4mtwcTayf4agsXRSVwcA-WJCG2odmMfw6UNgsUa-ObzWLkteBw",
                "consumer-id": "ecom",
                "correlation-id": "ecomr4-d093bbdf-ed26-4c58-b6b0-02ad72dc9cc6",
            },
        )
    except Exception as err:
        output = f"{type(err).__name__}: {str(err)}".encode()
    else:
        output = response.content

    with open(f"usfoods/{index:05d}-{item_number}.json", "wb") as file:
        file.write(output)

    time.sleep(1)
