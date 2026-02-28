import os
import time
import requests
import pdfplumber
from bs4 import BeautifulSoup

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}
DELAY = 1.5

HTML_URLS = {
    # Wikipedia Pittsburgh
    "wiki_pittsburgh": "https://en.wikipedia.org/wiki/Pittsburgh",
    "wiki_pittsburgh_history": "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
    "wiki_pittsburgh_culture": "https://en.wikipedia.org/wiki/Culture_of_Pittsburgh",
    "wiki_pittsburgh_geography": "https://en.wikipedia.org/wiki/Geography_of_Pittsburgh",
    "wiki_pittsburgh_neighborhoods": "https://en.wikipedia.org/wiki/List_of_Pittsburgh_neighborhoods",
    "wiki_pittsburgh_economy": "https://en.wikipedia.org/wiki/Economy_of_Pittsburgh",

    # Wikipedia CMU
    "wiki_cmu": "https://en.wikipedia.org/wiki/Carnegie_Mellon_University",
    "wiki_cmu_scs": "https://en.wikipedia.org/wiki/School_of_Computer_Science_at_Carnegie_Mellon_University",
    "wiki_cmu_tepper": "https://en.wikipedia.org/wiki/Tepper_School_of_Business",

    # Wikipedia Sports
    "wiki_pirates": "https://en.wikipedia.org/wiki/Pittsburgh_Pirates",
    "wiki_steelers": "https://en.wikipedia.org/wiki/Pittsburgh_Steelers",
    "wiki_penguins": "https://en.wikipedia.org/wiki/Pittsburgh_Penguins",
    "wiki_pitt_panthers": "https://en.wikipedia.org/wiki/Pittsburgh_Panthers",

    # Wikipedia Venues & Landmarks
    "wiki_pnc_park": "https://en.wikipedia.org/wiki/PNC_Park",
    "wiki_acrisure_stadium": "https://en.wikipedia.org/wiki/Acrisure_Stadium",
    "wiki_ppg_paints_arena":"https://en.wikipedia.org/wiki/PPG_Paints_Arena",
    "wiki_phipps": "https://en.wikipedia.org/wiki/Phipps_Conservatory_and_Botanical_Gardens",
    "wiki_duquesne_incline": "https://en.wikipedia.org/wiki/Duquesne_Incline",
    "wiki_point_state_park": "https://en.wikipedia.org/wiki/Point_State_Park",
    "wiki_andy_warhol_museum": "https://en.wikipedia.org/wiki/Andy_Warhol_Museum",
    "wiki_carnegie_museums": "https://en.wikipedia.org/wiki/Carnegie_Museums_of_Pittsburgh",
    "wiki_heinz_history": "https://en.wikipedia.org/wiki/Heinz_History_Center",
    "wiki_pgh_symphony": "https://en.wikipedia.org/wiki/Pittsburgh_Symphony_Orchestra",
    "wiki_pgh_opera": "https://en.wikipedia.org/wiki/Pittsburgh_Opera",
    "wiki_pgh_zoo": "https://en.wikipedia.org/wiki/Pittsburgh_Zoo_%26_Aquarium",
    "wiki_pgh_intl_airport": "https://en.wikipedia.org/wiki/Pittsburgh_International_Airport",

    # CMU Official
    "cmu_about": "https://www.cmu.edu/about/",
    "cmu_history":  "https://www.cmu.edu/about/history.html",
    "cmu_facts": "https://www.cmu.edu/about/cmu-quick-facts.html",
    "cmu_schools":  "https://www.cmu.edu/academics/",
    "cmu_research": "https://www.cmu.edu/research/index.html",

    # Visit Pittsburgh
    "visit_pgh_home":  "https://www.visitpittsburgh.com/",
    "visit_pgh_neighborhoods": "https://www.visitpittsburgh.com/neighborhoods/",
    "visit_pgh_arts": "https://www.visitpittsburgh.com/things-to-do/arts-culture/",
    "visit_pgh_sports": "https://www.visitpittsburgh.com/things-to-do/sports-outdoors/",
    "visit_pgh_history": "https://www.visitpittsburgh.com/things-to-do/history-heritage/",
    "visit_pgh_food": "https://www.visitpittsburgh.com/things-to-do/food-drink/",
    "visit_pgh_events": "https://www.visitpittsburgh.com/events/",

    # City of Pittsburgh
    "city_pgh_home": "https://pittsburghpa.gov/",
    "city_pgh_about": "https://pittsburghpa.gov/mayor/index.html",

    # Encyclopedia Britannica
    "britannica_pgh": "https://www.britannica.com/place/Pittsburgh",
    "britannica_cmu": "https://www.britannica.com/topic/Carnegie-Mellon-University",

    # Sports
    "pirates_home": "https://www.mlb.com/pirates",
    "steelers_home": "https://www.steelers.com/",
    "penguins_home": "https://www.nhl.com/penguins",

    # Music and Culture 
    "carnegie_museums": "https://carnegiemuseums.org/",
    "heinz_history_center": "https://www.heinzhistorycenter.org/",
    "pgh_symphony": "https://www.pittsburghsymphony.org/",
    "pgh_opera": "https://pittsburghopera.org/",
    "cultural_trust": "https://trustarts.org/",
    "frick_museum": "https://www.thefrickpittsburgh.org/",

    # Food-related events
    "picklesburgh": "https://www.picklesburgh.com/",
    "taco_fest": "https://www.pittsburghtacofest.com/",
    "restaurant_week":  "https://www.pittsburghrestaurantweek.com/",
    "banana_split_fest": "https://bananasplitfestival.com/",

    # Events
    "downtown_pgh_events": "https://www.downtownpittsburgh.com/events/",
    "cmu_events": "https://www.cmu.edu/engage/events/",
    "pgh_city_paper_events": "https://www.pghcitypaper.com/pittsburgh/EventsCalendar",
}

PDF_CANDIDATES = {"budget_2025": ["https://apps.pittsburghpa.gov/redtail/images/24295_2025_Operating_Budget_APPROVED.pdf",
        "https://apps.pittsburghpa.gov/redtail/images/23505_2025_Proposed_Operating_Budget.pdf",
        "https://pittsburghpa.gov/finance/budget",],
}

# Scraping
def scrape_html(name, url):
    save_path = os.path.join(SAVE_DIR, name + ".txt")
    if os.path.exists(save_path):
        print(f"[SKIP] {name}")
        return
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 30:
                lines.append(text)

        content = f"SOURCE: {url}\n\n" + "\n\n".join(lines)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] {name} — {len(content):,} chars")

    except Exception as e:
        print(f"[FAIL] {name} —{e}")

# Download PDF and extract text
def scrape_pdf(name, url):
    pdf_path = os.path.join(SAVE_DIR, name + ".pdf")
    txt_path = os.path.join(SAVE_DIR, name + ".txt")

    if os.path.exists(txt_path):
        print(f"[SKIP] {name}")
        return

    try:
        print(f"[DL] {name} —downloading...")
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages.append(f"[Page {i+1}]\n{text}")

        content = f"SOURCE: {url}\n\n" + "\n\n".join(pages)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[OK] {name} —{len(content):,} chars")

    except Exception as e:
        print(f"[FAIL] {name} —{e}")

if __name__ == "__main__":
    print(f"Saving to: {os.path.abspath(SAVE_DIR)}\n")

    print("HTML Pages：")
    for name, url in HTML_URLS.items():
        scrape_html(name, url)
        time.sleep(DELAY)

    print("\nMonthly Events：")
    months = {"march": "03", "april": "04", "may": "05", "june": "06",
        "july": "07", "august": "08", "september": "09", "october": "10"}
    for month_name, month_code in months.items():
        url = f"https://www.visitpittsburgh.com/events/?date=2026-{month_code}"
        scrape_html(f"visit_pgh_events_{month_name}", url)
        time.sleep(DELAY)

    print("\nPDFs：")
    for name, candidates in PDF_CANDIDATES.items():
        for url in candidates:
            result = scrape_pdf(name, url)
            if result:
                break
            time.sleep(DELAY)
        time.sleep(DELAY)

    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]
    print(f"\nDone! {len(files)} files saved to {SAVE_DIR}/")