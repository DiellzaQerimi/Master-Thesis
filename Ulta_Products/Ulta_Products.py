from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import pandas as pd

driver = webdriver.Chrome()
driver.get("https://www.ulta.com/shop/skin-care/all")
wait = WebDriverWait(driver, 20)

product_selector = 'a.pal-c-Link--primary[href*="/p/"]'
all_links = set()
productLinks = []

# Collects all currently loaded product links from the page
def get_all_product_links():
    elements = driver.find_elements(By.CSS_SELECTOR, product_selector)
    return {el.get_attribute("href") for el in elements if el.get_attribute("href")}

# Scrolls down gradually to trigger lazy loading until the page height stops changing
def smooth_scroll_to_bottom(step=300, pause=0.7, max_no_change=3):
    no_change_count = 0
    last_height = driver.execute_script("return document.body.scrollHeight")

    while no_change_count < max_no_change:
        driver.execute_script(f"window.scrollBy(0, {step});")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            no_change_count += 1
        else:
            no_change_count = 0
            last_height = new_height

# Loads the initial set of products by scrolling and collecting visible links
smooth_scroll_to_bottom()
all_links.update(get_all_product_links())
print(f"Initially loaded products: {len(all_links)}")

# Continuously clicks the load button to retrieve more products until nothing new is loaded
while True:
    smooth_scroll_to_bottom()

    try:
        show_more = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.LoadContent__button'))
        )

        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});", show_more
        )
        driver.execute_script("arguments[0].click();", show_more)

        WebDriverWait(driver, 15).until(
            lambda d: len(get_all_product_links()) > len(all_links)
        )
        all_links.update(get_all_product_links())
        print(f"Loaded {len(all_links)} products so far")

    except TimeoutException:
        print("No more button, or nothing new loaded.")
        break

print(f"\nDone! Total products found: {len(all_links)}")

# Extracts Product IDs from product URLs and stores them alongside their links
for link in all_links:
    match = re.search(r'(pimprod\d+)', link)
    if match:
        pid = match.group(1)
        productLinks.append({"Link": link, "Product ID": pid})

# Saves the collected product links and IDs into a CSV file
df = pd.DataFrame(productLinks)
df = df.drop_duplicates(subset="Product ID")  # remove duplicates
csv_file = "Ulta_Products1.csv"
df.to_csv(csv_file, index=False)
print(f"Saved to {csv_file}")

driver.quit()
