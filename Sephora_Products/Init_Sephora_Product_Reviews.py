import requests
import csv
import time

PASSKEY = "calXm2DyQVjcCy9agq85vmTJv5ELuuBCF2sdg4BnJzJus"
LOCALE = "en_US"
LIMIT = 100  # max reviews per request
INPUT_CSV = "Sephora_Products.csv"   # input file with product IDs  
OUTPUT_CSV = "Sephora_Product_Reviews.csv"  # output file with product reviews

def fetch_user_reviews(product_id):
    offset = 0
    total = 1
    user_reviews = []

    while offset < total:
        url = (
            f"https://api.bazaarvoice.com/data/reviews.json?"
            f"Filter=ProductId:{product_id}"
            f"&Sort=SubmissionTime:desc"
            f"&Limit={LIMIT}"
            f"&Offset={offset}"
            f"&Stats=Reviews"
            f"&passkey={PASSKEY}"
            f"&apiversion=5.4"
            f"&Locale={LOCALE}"
        )

        response = requests.get(url)
        data = response.json()

        total = data.get("TotalResults", 0)

        results = data.get("Results", [])

        for review in results:
            context = review.get("ContextDataValues", {})
            raw_review_text = review.get("ReviewText") or ""
            clean_review_text = raw_review_text.replace("\n", " ").replace("\r", " ").strip()

            user_reviews.append({
                "product_id": product_id,
                "product_name": review.get("OriginalProductName", ""),
                "submission_time": review.get("SubmissionTime", ""),
                "title": review.get("Title", ""),
                "review_text": clean_review_text,
                "rating": review.get("Rating", ""),
                "skin_tone": context.get("skinTone", {}).get("ValueLabel", ""),
                "skin_type": context.get("skinType", {}).get("ValueLabel", ""),
                "age": context.get("age", {}).get("ValueLabel", ""),
                "hair_color": context.get("hairColor", {}).get("ValueLabel", ""),
                "eye_color": context.get("eyeColor", {}).get("ValueLabel", ""),
            })


        offset += LIMIT
        time.sleep(0.2)

    return user_reviews

def main():
    all_user_reviews = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            product_id = row.get("Product ID")
            if product_id:
                print(f"\nFetching reviews for {product_id}...")
                user_reviews = fetch_user_reviews(product_id)
                print(f"{len(user_reviews)} reviews found for {product_id}")
                all_user_reviews.extend(user_reviews)

    if all_user_reviews:
        keys = all_user_reviews[0].keys()
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_user_reviews)

        print(f"\n Saved {len(all_user_reviews)} user reviews to '{OUTPUT_CSV}'")
    else:
        print("\nNo user reviews found to save.")

if __name__ == "__main__":
    main()
