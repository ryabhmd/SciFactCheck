"""
Script to crawl IEP (https://iep.utm.edu/) to gather entities from articles to represent the Arts and Humanities field.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

IEP_INFO = {
    "history_of_philosophy": {
        "NAME": "History of Philosophy",
        "BASE_URL": "https://iep.utm.edu/category/history/page/",
        "NUM_PAGES": 8
    },
    "metaphysics_epistemology": {
        "NAME": "Metaphysics and Epistemology",
        "BASE_URL": "https://iep.utm.edu/category/m-and-e/page/", 
        "NUM_PAGES": 7
    },
    "philosophical_traditions": {
        "NAME": "Philosophical Traditions",
        "BASE_URL": "https://iep.utm.edu/category/traditions/page/", 
        "NUM_PAGES": 3
    },
    "science_logic_math": {
        "NAME": "Science, Logic, and Mathematics",
        "BASE_URL": "https://iep.utm.edu/category/s-l-m/page/", 
        "NUM_PAGES": 3
    },
    "value_theory": {
        "NAME": "Value Theory",
        "BASE_URL": "https://iep.utm.edu/category/value/page/", 
        "NUM_PAGES": 4
    }
}

def scrape_iep_category(base_url: str, num_pages: int, delay: float = 1.0) -> pd.DataFrame:
    """
    Scrape article titles and URLs from IEP category pages and return a pandas DataFrame.

    Args:
        base_url (str): Example: "https://iep.utm.edu/category/history/page/"
        num_pages (int): Number of pages to scrape.
        delay (float): Optional delay between requests (in seconds).

    Returns:
        pd.DataFrame: Columns = ["title", "url"]
    """
    records = []

    for page in range(1, num_pages + 1):
        url = f"{base_url}{page}/"
        print(f"Scraping page {page}: {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "content"})
        if not content_div:
            print(f"No content found on page {page}")
            continue

        for ul in content_div.find_all("ul"):
            a_tag = ul.find("a", href=True)
            if a_tag:
                records.append({
                    "title": a_tag.get_text(strip=True),
                    "url": a_tag["href"]
                })

        time.sleep(delay)

    df = pd.DataFrame(records)
    return df

def extract_iep_article_sections(url: str) -> dict:
    """
    Extracts the full text of an IEP article organized by section headers.

    Args:
        url (str): IEP article URL (e.g., 'https://iep.utm.edu/kant-rel/').

    Returns:
        dict: {
            "title": <article title>,
            "sections": {<section title>: <section text>, ...},
            "url": <url>
        }
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"title": None, "sections": {}, "url": url}

    soup = BeautifulSoup(response.text, "html.parser")

    entry_div = soup.find("div", class_="entry-content")
    if not entry_div:
        print(f"No entry-content found on {url}")
        return {"title": None, "sections": {}, "url": url}

    # Article title
    title_tag = entry_div.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Clean up unwanted tags
    for tag in entry_div.find_all(["img", "table", "ul", "ol", "nav", "figure"]):
        tag.decompose()

    # --- Parse sections ---
    sections = {}
    current_section = "Introduction"
    section_text = []

    # iterate through children, capturing text under each heading
    for elem in entry_div.find_all(["p", "h2", "h3"]):
        if elem.name.startswith("h"):
            # new section starts
            if section_text:
                sections[current_section] = "\n".join(section_text).strip()
                section_text = []
            current_section = elem.get_text(" ", strip=True)
        elif elem.name == "p":
            text = elem.get_text(" ", strip=True)
            if text:
                section_text.append(text)

    # save the last section
    if section_text:
        sections[current_section] = "\n".join(section_text).strip()

    return {"title": title, "sections": sections, "url": url}

def main():
    
    # Gather IEP data per category
    all_topics_data = []
    for topic, info in IEP_INFO.items():
        data = scrape_iep_category(info["BASE_URL"], info["NUM_PAGES"])
        data['topic'] = [info["Name"]] * len(data)
        all_topics_data.append(data)

    all_topics_data = [df.reset_index(drop=True) for df in all_topics_data]

    # For each topic, sample 100 unique rows that aren’t already in the global set
    final_df = pd.DataFrame()
    used_urls = set()
    
    for i, df in enumerate(all_topics_data):
        # Drop duplicate URLs within the same topic
        df = df.drop_duplicates(subset="url")
        
        # Filter out rows whose URL already exists in final_df
        df = df[~df["url"].isin(used_urls)]
        
        # Sample up to 100 unique rows
        n_to_sample = min(100, len(df))
        sampled = df.sample(n=n_to_sample, random_state=42)
        
        # Add sampled rows to the final DataFrame
        final_df = pd.concat([final_df, sampled], ignore_index=True)
        
        # Update the set of used URLs
        used_urls.update(sampled["url"])
    
    # Double-check final deduplication
    final_df = final_df.drop_duplicates(subset="url", ignore_index=True)
    # Get full-text article
    final_df['full_text'] = final_df['url'].apply(extract_iep_article_sections)

    final_df.to_pickle('iep_data_with_full_text.pkl')

if __name__ == '__main__':
 	main()


