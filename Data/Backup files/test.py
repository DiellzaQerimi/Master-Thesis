from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time, re, pandas as pd

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
START_URL      = "https://www.ulta.com/shop/skin-care/all?page=70"
SCROLL_STEP_PX = 300
SCROLL_PAUSE_S = 0.7
BACKUP_EVERY   = 200                 # save a CSV every N links
OUTPUT_FILE    = "ulta_links9.csv"

# ─────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────
driver = webdriver.Chrome()
wait   = WebDriverWait(driver, 20)
driver.get(START_URL)

product_sel = 'a.pal-c-Link--primary[href*="/p/"]'
all_links   = set()                  # href strings
records     = []                     # dicts for CSV rows

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def smooth_scroll_to_bottom(step=SCROLL_STEP_PX, pause=SCROLL_PAUSE_S):
    """Scroll smoothly to the current bottom of the page."""
    last_h = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script(f"window.scrollBy(0,{step});")
        time.sleep(pause)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break
        last_h = new_h

def get_all_product_links() -> set[str]:
    return {
        el.get_attribute("href")
        for el in driver.find_elements(By.CSS_SELECTOR, product_sel)
        if el.get_attribute("href")
    }

def get_load_more_button():
    """
    Return the **bottom‑most** button whose text is 'Load More'.
    If none exists, return None.
    """
    btns = driver.find_elements(By.CSS_SELECTOR, "button.LoadContent__button")
    for btn in reversed(btns):            # walk from bottom up
        if btn.text.strip().lower() == "load more":
            return btn
    return None

def backup_csv(counter: int):
    df = pd.DataFrame({"Link": list(all_links)})
    fname = f"ulta_backup_{counter}.csv"
    df.to_csv(fname, index=False)
    print(f"⟲ Backup → {fname}")

# ─────────────────────────────────────────
# PRIME PAGE
# ─────────────────────────────────────────
smooth_scroll_to_bottom()
all_links.update(get_all_product_links())
print(f"Initially loaded: {len(all_links)} products")

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:
    smooth_scroll_to_bottom()             # make sure button is in DOM
    try:
        btn = wait.until(lambda d: get_load_more_button())
        if btn is None:
            print("No 'Load More' button left — scraping finished.")
            break

        # scroll into view & JS‑click (avoids overlays)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        driver.execute_script("arguments[0].click();", btn)

        # wait until we actually have new links
        prev = len(all_links)
        WebDriverWait(driver, 15).until(
            lambda d: len(get_all_product_links()) > prev
        )

        all_links.update(get_all_product_links())
        print(f"Loaded {len(all_links)} products so far")

        # periodic backup
        if len(all_links) % BACKUP_EVERY == 0:
            backup_csv(len(all_links))

    except TimeoutException:
        print("Timed out waiting for new items — assuming no more pages.")
        break

# ─────────────────────────────────────────
# SAVE FINAL CSV
# ─────────────────────────────────────────
for link in all_links:
    if m := re.search(r"(pimprod\d+)", link):
        records.append({"Link": link, "Product ID": m.group(1)})

pd.DataFrame(records).to_csv(OUTPUT_FILE, index=False)
print(f"\n🎉 Done! {len(records)} products saved to {OUTPUT_FILE}")

driver.quit()
