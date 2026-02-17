import csv
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

seen_ids = set()

# Safely retrieves text from a single element, returning a default if not found
def safe_get_text(driver, by, selector, default=""):
    try:
        return driver.find_element(by, selector).text.strip()
    except NoSuchElementException:
        return default
    except Exception:
        return default

# Expands an accordion section by its aria-label and returns the extracted content text
def expand_section_and_get_text(driver, section_title):
    try:
        # Find the button using aria-label
        button_xpath = f'//button[@class="pal-c-Accordion__button" and @aria-label="{section_title}"]'
        button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, button_xpath))
        )

        # Scroll into view
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", button)
        time.sleep(1)

        # Click with JS (bypasses overlays or sticky headers)
        driver.execute_script("arguments[0].click();", button)
        time.sleep(1)

        # Get the content div (the next sibling of the button’s parent)
        parent = button.find_element(By.XPATH, "./..")
        content = parent.find_element(By.XPATH, './following-sibling::*[1]')
        return content.text.strip()

    except Exception as e:
        print(f"Could not get section '{section_title}': {e}")
        return ""

# Scrapes structured product fields from an Ulta product page and returns them as a dictionary
def scrape_product(driver, url, product_id):
    driver.get(url)
    time.sleep(2)

    data = {
        "product_id": product_id,
        "brand": "",
        "product": "",
        "category": "",
        "subcategory": "",
        "price": "",
        "size": "",
        "image": "",
        "no_of_reviews": "",
        "description": "",
        "about_the_product": "",
        "ingredients": "",
        "how_to_use": ""
    }

    try:
        wait = WebDriverWait(driver, 15)

        try:
            data["brand"] = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1 a span.pal-c-Link__label"))).text.strip()
        except Exception:
            pass

        data["product"] = safe_get_text(driver, By.CSS_SELECTOR, 'h1 span.Text-ds--title-5')
        data["description"] = safe_get_text(driver, By.CSS_SELECTOR, 'div p.Text-ds--subtitle-1')
        
        try:
            data["category"] = driver.find_element(By.XPATH, '(//ul[@id="Breadcrumbs__List"]/li)[3]').text.strip()
            data["subcategory"] = driver.find_element(By.XPATH, '(//ul[@id="Breadcrumbs__List"]/li)[4]').text.strip()
        except Exception:
            pass

        ProductPricing = driver.find_element(By.CSS_SELECTOR, 'div.ProductPricing')
        try:
            data["price"] = ProductPricing.find_element(By.CSS_SELECTOR, 'span.Text-ds.Text-ds--title-5.Text-ds--left.Text-ds--black').text.strip()
        except Exception:
            data["price"] = ProductPricing.find_element(By.CSS_SELECTOR, 'span.Text-ds.Text-ds--body-3.Text-ds--left.Text-ds--neutral-600.Text-ds--line-through').text.strip()

        try:
            variant = driver.find_element(By.CSS_SELECTOR, 'div.ProductVariant')
            dimensions = variant.find_elements(By.CSS_SELECTOR, 'div.ProductDimension')
            for dim in dimensions:
                label = dim.find_element(By.CSS_SELECTOR, 'span.Text-ds--neutral-600').text.strip()
                if label == 'Size:':
                    data["size"] = dim.find_element(By.CSS_SELECTOR, 'span.Text-ds--black').text.strip()
                    break        
        except Exception:
            pass

        Image = driver.find_element(By.CSS_SELECTOR, 'div.CarouselMobile--Image')
        try:
            data["image"] = Image.find_element(By.CSS_SELECTOR, 'img').get_attribute("src")
        except Exception:
            data["image"] = ""

        data["no_of_reviews"] = safe_get_text(driver, By.CSS_SELECTOR, "a[href='#reviews'] span.pal-c-Link__label span")

        data["about_the_product"] = expand_section_and_get_text(driver, "Details")
        data["ingredients"] = expand_section_and_get_text(driver, "Ingredients")
        data["how_to_use"] = expand_section_and_get_text(driver, "How To Use")

    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return data

# Reads product IDs and URLs from the input CSV, scrapes each product page, and writes results to an output CSV
input_file = "Data/Ulta_Products.csv"
output_file = "Ulta_Product_Details.csv"

driver = webdriver.Chrome()
results = []

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        product_id = row["Product ID"]
        url = row["Link"]
        if product_id in seen_ids:
            continue
        seen_ids.add(product_id)
        print(f"Scraping {product_id}...")
        result = scrape_product(driver, url, product_id)
        results.append(result)
        time.sleep(2)

driver.quit()

# Saves the scraped product details into a structured CSV file
with open(output_file, mode='w', newline='', encoding='utf-8-sig') as csvfile:
    fieldnames = [
        "product_id", "brand", "product", "category", "subcategory",
        "price", "size", "image", "no_of_reviews", "description", "about_the_product",
        "ingredients", "how_to_use"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. {len(results)} products saved to {output_file}")
