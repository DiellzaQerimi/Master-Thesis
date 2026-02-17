import csv
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Closes the initial modal popup if it appears
def dismiss_popup(driver):
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-at="modal_close"]'))
        ).click()
    except TimeoutException:
        pass

# Closes the sign-in prompt popup if it appears
def dismiss_sign_in(driver):
    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-at="close_button"]'))
        ).click()
    except TimeoutException:
        pass

# Safely retrieves text from a single element, returning a default if not found
def safe_get_text(driver, by, selector, default=""):
    try:
        return driver.find_element(by, selector).text.strip()
    except NoSuchElementException:
        return default
    except Exception:
        return default

# Expands a collapsible section by title and returns its visible text content
def expand_and_get_text(driver, section_title, content_id):
    try:
        header = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f'//h2[contains(text(), "{section_title}")]'))
        )
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", header)
        time.sleep(1)

        button = driver.find_element(By.XPATH, f'//button[h2[contains(text(), "{section_title}")]]')
        driver.execute_script("arguments[0].click();", button)
        time.sleep(2)

        content_div = driver.find_element(By.ID, content_id)
        return content_div.text.strip()
    except Exception:
        return ""

# Scrapes structured product fields from a Sephora product page and returns them as a dictionary
def scrape_product(driver, url, product_id):
    driver.get(url)
    time.sleep(2)

    dismiss_popup(driver)
    dismiss_sign_in(driver)

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
        "about_the_product": "",
        "ingredients": "",
        "how_to_use": ""
    }

    try:
        wait = WebDriverWait(driver, 15)

        try:
            data["brand"] = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-at="brand_name"]'))
            ).text.strip()
        except Exception:
            pass

        data["product"] = safe_get_text(driver, By.CSS_SELECTOR, 'span.css-wkag1e')

        try:
            breadcrumbs = driver.find_elements(By.CSS_SELECTOR, 'a.css-d747d0')
            data["category"] = breadcrumbs[1].text if len(breadcrumbs) > 1 else ""
            data["subcategory"] = breadcrumbs[2].text if len(breadcrumbs) > 2 else ""
        except Exception:
            pass
        
        data["price"] = safe_get_text(driver, By.CSS_SELECTOR, 'b.css-0')

        if data["price"] == "":
            data["price"] = safe_get_text(driver, By.CSS_SELECTOR, 'b.css-p9xrit')
        else:
            pass

        data["size"] = safe_get_text(driver, By.CSS_SELECTOR, 'span.css-15ro776')
        if not data["size"] or "Color" in data["size"] or "Size" not in data["size"] or "oz" not in data["size"]:
            data["size"] = safe_get_text(driver, By.CSS_SELECTOR, 'span[data-at="sku_size_label"]').replace("Size: ", "")
            if not data["size"]:
                data["size"] = safe_get_text(driver, By.CSS_SELECTOR, 'span.css-1nxxl34').replace("Size: ", "")
        else:
            data["size"] = data["size"].replace("Size: ", "")

        if any(x in data["size"].lower() for x in ["payment", "klarna", "afterpay"]):
            data["size"] = ""

        try:
            data["image"] = driver.find_element(By.CSS_SELECTOR, 'img.css-tl1r8e.e15t7owz0').get_attribute("src")
        except Exception:
            data["image"] = ""

        data["no_of_reviews"] = safe_get_text(driver, By.CSS_SELECTOR, 'a.css-137xvot')

        # Expands the main content section if a "Show more" button is present
        try:
            show_more = driver.find_element(By.XPATH, '//button[text()="Show more"]')
            driver.execute_script("arguments[0].scrollIntoView();", show_more)
            time.sleep(1)
            show_more.click()
            time.sleep(1)
        except NoSuchElementException:
            pass

        # Extracts the "About the Product" text by collecting sibling blocks until the next section header
        try:
            about_header = driver.find_element(By.XPATH, '//h2[contains(text(), "About the Product")]')
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", about_header)
            time.sleep(1)

            about_text_parts = []
            sibling = about_header.find_element(By.XPATH, 'following-sibling::*[1]')
            while sibling:
                if sibling.tag_name == 'h2':
                    break
                text = sibling.text.strip()
                if text:
                    about_text_parts.append(text)
                try:
                    sibling = sibling.find_element(By.XPATH, 'following-sibling::*[1]')
                except:
                    break

            data["about_the_product"] = "\n".join(about_text_parts)
        except Exception:
            data["about_the_product"] = ""

        data["ingredients"] = expand_and_get_text(driver, "Ingredients", "ingredients")
        data["how_to_use"] = expand_and_get_text(driver, "How to Use", "howtouse")

    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return data

# Reads product IDs and URLs from the input CSV, scrapes each product page, and writes results to an output CSV
input_file = "Sephora_Products.csv"
output_file = "Sephora_Product_Details.csv"

driver = webdriver.Chrome()
results = []

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        product_id = row["Product ID"]
        url = row["Link"]
        print(f"Scraping {product_id}...")
        result = scrape_product(driver, url, product_id)
        results.append(result)

driver.quit()

# Saves the scraped product details into a structured CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        "product_id", "brand", "product", "category", "subcategory",
        "price", "size", "image", "no_of_reviews", "about_the_product",
        "ingredients", "how_to_use"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. {len(results)} products saved to {output_file}")
