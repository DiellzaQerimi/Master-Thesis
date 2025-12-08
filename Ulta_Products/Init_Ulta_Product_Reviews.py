import requests
import csv
import time

INPUT_CSV = "Ulta_Products.csv"
OUTPUT_CSV = "Init_Ulta_Product_Reviews.csv"
BASE_URL = "https://display.powerreviews.com"
API_KEY = "daa0f241-c242-4483-afb7-4449942d1a2b"
PAGE_SIZE = 24

def fetch_user_reviews(product_id):
    user_reviews = []
    offset = 0  # PowerReviews uses 1-based indexing
    total_reviews = None
    page = 1

    while True:
        url = (
            f"{BASE_URL}/m/6406/l/en_US/product/{product_id}/reviews"
            f"?paging.from={offset}&paging.size={PAGE_SIZE}"
            f"&filters=&search=&sort=Newest&image_only=false&page_locale=en_US"
            f"&_noconfig=true&apikey={API_KEY}"
        )

        try:
            response = requests.get(url)
            data = response.json()
        except Exception as e:
            print(f"Error fetching data for {product_id}: {e}")
            break

        if total_reviews is None:
            total_reviews = data.get("paging", {}).get("total_results", 0)
            if total_reviews == 0:
                break

        results = data.get("results", [])
        if not results or not results[0].get("reviews"):
            break

        reviews = results[0]["reviews"]

        for review in reviews:
            details = review.get("details", {})
            metrics = review.get("metrics", {})

            raw_review_text = details.get("comments", "")
            clean_review_text = raw_review_text.replace("\n", " ").replace("\r", " ").strip()

            user_reviews.append({
                "product_id": product_id,
                "submission_time": details.get("created_date", ""),
                "title": details.get("headline", ""),
                "review_text": clean_review_text,
                "rating": metrics.get("rating", ""),
                "location": details.get("location", ""),
            })

        offset += PAGE_SIZE
        page += 1
        time.sleep(0.2)

        if offset > total_reviews:
            break

    return user_reviews

def main():
    all_user_reviews = []

    try:
        with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                product_id = row.get("Product ID") or row.get("product_id")
                if product_id:
                    print(f"\nFetching reviews for {product_id}...")
                    user_reviews = fetch_user_reviews(product_id)
                    print(f"{len(user_reviews)} total reviews found for {product_id}")
                    all_user_reviews.extend(user_reviews)
    except FileNotFoundError:
        print(f"Input file '{INPUT_CSV}' not found.")
        return

    if all_user_reviews:
        keys = all_user_reviews[0].keys()
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_user_reviews)
        print(f"\n Saved {len(all_user_reviews)} total reviews to '{OUTPUT_CSV}'")
    else:
        print("\n No reviews found to save.")

if __name__ == "__main__":
    main()
