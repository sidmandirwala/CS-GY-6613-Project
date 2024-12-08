from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import logging
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "medium_scraper"
COLLECTION_NAME = "repositories"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MediumCrawler:
    def __init__(self):
        # Set up Chrome options
        self.options = Options()
        self.options.add_argument("--headless")  # run in headless mode

        # Use ChromeDriverManager with Service
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=self.options)

    def scroll_page(self, scroll_pause_time=1):
        """Scroll down the page to load more content."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to the bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)

            # Check if we've reached the bottom
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def extract(self, link: str):
        logger.info(f"Starting to scrape Medium article: {link}")

        self.driver.get(link)
        self.scroll_page()

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

        data = {
            "Title": title[0].get_text(strip=True) if title else None,
            "Subtitle": subtitle[0].get_text(strip=True) if subtitle else None,
            "Content": soup.get_text(strip=True),
        }

        logger.info(f"Successfully scraped article: {link}")
        return data

    def close(self):
        """Close the driver after scraping is done."""
        self.driver.quit()


# Example usage:
crawler = MediumCrawler()
article_link = "https://medium.com/schmiedeone/getting-started-with-ros2-part-1-d4c3b7335c71"  # Replace with a valid Medium article URL
data = crawler.extract(article_link)
print(data)
crawler.close()  # Close the driver once done

import time
import logging
import uuid
from pymongo import MongoClient, errors
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "medium_scraper"
COLLECTION_NAME = "repositories"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MediumCrawler:
    def __init__(self):
        # Set up Chrome options
        self.options = Options()
        self.options.add_argument("--headless")  # run in headless mode

        # Use ChromeDriverManager with Service
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=self.options)

    def scroll_page(self, scroll_pause_time=1):
        """Scroll down the page to load more content."""
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to the bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)

            # Check if we've reached the bottom
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def extract(self, link: str, user: dict = None):
        # Check if the article already exists in the database
        if collection.find_one({"link": link}):
            logger.info(f"Article already exists in the database: {link}")
            return

        logger.info(f"Starting to scrape Medium article: {link}")

        self.driver.get(link)
        self.scroll_page()

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

        data = {
            "Title": title[0].get_text(strip=True) if title else None,
            "Subtitle": subtitle[0].get_text(strip=True) if subtitle else None,
            "Content": soup.get_text(strip=True),
        }

        # Save the article data to MongoDB
        doc = {
            "_id": str(uuid.uuid4()),
            "link": link,
            "platform": "medium",
            "content": data,
        }

        if user:
            doc["author_id"] = user["id"]
            doc["author_full_name"] = user["full_name"]

        try:
            collection.insert_one(doc)
            logger.info(f"Successfully scraped and saved article: {link}")
        except errors.PyMongoError as e:
            logger.error(f"Failed to save article to MongoDB: {e}")

        return data

    def close(self):
        """Close the driver after scraping is done."""
        self.driver.quit()

# Example usage:
crawler = MediumCrawler()
test_user = {"id": str(uuid.uuid4()), "full_name": "Test User"}
article_link = "https://medium.com/schmiedeone/getting-started-with-ros2-part-1-d4c3b7335c71"
data = crawler.extract(article_link, user=test_user)
print(data)
crawler.close()