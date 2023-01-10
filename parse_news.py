from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def parse_bbc_news(base_url, page_path):
    # Use sync version of Playwright
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch()

        # Open a new browser page
        page = browser.new_page()

        # Open our test file in the opened page
        page.goto(page_path)
        page_content = page.content()

        # Process extracted content with BeautifulSoup
        soup = BeautifulSoup(page_content, features="html.parser")

        all_articles = soup.find_all(attrs={'class': "gel-layout__item"})
        to_extract = {}
        for article in all_articles:
            links = article.find_all(name='a')
            if links is None or len(links) <= 1:
                continue
            href = links[0].attrs.get('href')

            metadata = article.find_all(name='li', attrs={'class': 'nw-c-promo-meta'})
            if metadata is None or len(metadata) <= 1:
                continue
            to_extract[href] = article.get_text()

        results = {}
        for i, (href, text) in enumerate(to_extract.items()):
            final_link = base_url + href
            page.goto(final_link)
            page_content = page.content()
            soup = BeautifulSoup(page_content, features="html.parser")
            paragraphs = soup.find_all(attrs={"data-component": "text-block"})
            text = "\n".join([p.get_text() for p in paragraphs])

            results[final_link] = text

        # Close browser
        browser.close()

    return results

