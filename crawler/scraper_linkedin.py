import time
import uuid
from typing import Dict
from bs4 import BeautifulSoup
from loguru import logger
from pymongo import MongoClient, errors
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service  # Updated import
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager  # To manage ChromeDriver

# Configure Loguru
logger.add("linkedin_scraper.log", rotation="1 MB")  # Logs to a file with rotation

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "linkedin_scraper"
COLLECTION_NAME = "profiles"

client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

class LinkedInCrawler:
    def __init__(self, ignore=None, scroll_limit=5, headless=True) -> None:
        """Initializes the LinkedIn Crawler"""
        if ignore is None:
            ignore = (".git", ".toml", ".lock", ".png")
        
        self._ignore = ignore
        self._scroll_limit = scroll_limit
        self._headless = headless

        self.driver = self._initialize_driver()

    def _initialize_driver(self):
        """Initializes the Selenium WebDriver using ChromeDriverManager."""
        options = Options()
        if self._headless:
            options.add_argument("--headless=new")  # Updated headless argument for newer Chrome
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Set a realistic user-agent
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/115.0.0.0 Safari/537.36")
        
        # Initialize the Service object
        service = Service(ChromeDriverManager().install())
        
        # Initialize the WebDriver with the Service object
        driver = webdriver.Chrome(service=service, options=options)
        driver.maximize_window()
        return driver

    def login(self) -> None:
        """Logs into LinkedIn."""
        # Hardcoded credentials (⚠️ Not recommended for production use)
        username = "ai-project-1975b5340"  # Replace with your LinkedIn username
        password = "AI_Test"  # Replace with your LinkedIn password

        if not username or not password:
            raise ValueError("LinkedIn username or password is missing. Please check your hardcoded credentials.")
        
        logger.info("Navigating to LinkedIn login page.")
        self.driver.get("https://www.linkedin.com/login")
        
        try:
            # Wait until the username field is present
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            logger.info("Entering username.")
            self.driver.find_element(By.ID, "username").send_keys(username)
            logger.info("Entering password.")
            self.driver.find_element(By.ID, "password").send_keys(password)
            logger.info("Submitting login form.")
            self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
            
            # Wait for the home page to load by checking the presence of the profile avatar
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "profile-nav-item"))
            )
            logger.info("Logged in to LinkedIn successfully.")
        except Exception as e:
            logger.error(f"An error occurred during login: {e}")
            self.driver.save_screenshot("login_error.png")
            raise

    def extract(self, link: str, user: Dict) -> None:
        """Extracts content from a LinkedIn profile and saves it to MongoDB."""
        # Check if profile already exists
        if collection.find_one({"link": link}):
            logger.info(f"Profile already exists in the database: {link}")
            return

        logger.info(f"Starting to scrape LinkedIn profile: {link}")

        try:
            # Ensure we're logged in before scraping
            self.login()

            # Navigate to the profile page
            self.driver.get(link)
            
            # Wait until the profile name is loaded
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.text-heading-xlarge"))
            )
            logger.info("Profile page loaded successfully.")

            # Scroll to load dynamic content
            self._scroll_page()

            # Get profile page content
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            data = {
                "Name": self._scrape_section(soup, "h1", class_="text-heading-xlarge"),
                "About": self._scrape_about(soup),
                "Experience": self._scrape_experience(soup),
                "Education": self._scrape_education(soup),
                "Posts": self._scrape_posts(soup)
            }

            # Save the scraped data to MongoDB
            profile_data = {
                "_id": str(uuid.uuid4()),
                "name": data["Name"],
                "link": link,
                "about": data["About"],
                "experience": data["Experience"],
                "education": data["Education"],
                "posts": data["Posts"],
                "platform": "linkedin",
                "author_id": user["id"],
                "author_full_name": user["full_name"]
            }
            
            try:
                collection.insert_one(profile_data)
                logger.info(f"Profile data for {data['Name']} saved successfully.")
            except errors.PyMongoError as e:
                logger.error(f"Failed to save profile data to MongoDB: {e}")

            logger.info(f"Finished scraping LinkedIn profile: {link}")
        
        except Exception as e:
            logger.error(f"An error occurred while scraping the profile: {e}")
            self.driver.save_screenshot("scrape_error.png")

    def _scrape_section(self, soup: BeautifulSoup, tag: str, class_: str) -> str:
        """Scrapes a specific section of the LinkedIn profile."""
        parent_div = soup.find(tag, class_=class_)
        return parent_div.get_text(strip=True) if parent_div else ""

    def _scrape_about(self, soup: BeautifulSoup) -> str:
        """Scrapes the About section of the LinkedIn profile."""
        about_section = soup.find("section", {"id": "about"})
        if about_section:
            paragraphs = about_section.find_all("p")
            about_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
            return about_text
        return ""

    def _scrape_experience(self, soup: BeautifulSoup) -> str:
        """Scrapes the Experience section of the LinkedIn profile."""
        experience_section = soup.find("section", {"id": "experience-section"})
        if experience_section:
            experiences = experience_section.find_all("li")
            experience_text = ""
            for exp in experiences:
                title = exp.find("h3", class_="t-16 t-black t-bold")
                company = exp.find("span", class_="t-14 t-black--light t-normal")
                duration = exp.find("h4", class_="t-14 t-black--light t-normal")
                exp_details = []
                if title:
                    exp_details.append(f"Title: {title.get_text(strip=True)}")
                if company:
                    exp_details.append(f"Company: {company.get_text(strip=True)}")
                if duration:
                    exp_details.append(f"Duration: {duration.get_text(strip=True)}")
                experience_text += "; ".join(exp_details) + "\n"
            return experience_text.strip()
        return ""

    def _scrape_education(self, soup: BeautifulSoup) -> str:
        """Scrapes the Education section of the LinkedIn profile."""
        education_section = soup.find("section", {"id": "education-section"})
        if education_section:
            educations = education_section.find_all("li")
            education_text = ""
            for edu in educations:
                school = edu.find("h3", class_="pv-entity__school-name")
                degree = edu.find("span", class_="pv-entity__comma-item")
                duration = edu.find("p", class_="pv-entity__dates pv-entity__comma-item")
                edu_details = []
                if school:
                    edu_details.append(f"School: {school.get_text(strip=True)}")
                if degree:
                    edu_details.append(f"Degree: {degree.get_text(strip=True)}")
                if duration:
                    edu_details.append(f"Duration: {duration.get_text(strip=True)}")
                education_text += "; ".join(edu_details) + "\n"
            return education_text.strip()
        return ""

    def _scrape_posts(self, soup: BeautifulSoup) -> Dict[str, Dict[str, str]]:
        """Scrapes posts from the LinkedIn profile."""
        # Note: Scraping posts might require navigating to the 'Activity' section
        # Here, we'll attempt to scrape recent posts if available on the profile page

        posts = []
        activity_section = soup.find("section", {"id": "activity-section"})
        if activity_section:
            post_elements = activity_section.find_all("div", class_="ember-view")
            for i, post_element in enumerate(post_elements):
                post_text = post_element.get_text(separator="\n", strip=True)
                posts.append({"text": post_text})
        
        return {f"Post_{i}": post for i, post in enumerate(posts)}

    def _scroll_page(self):
        """Scrolls through the profile page to load dynamic content."""
        logger.info("Scrolling through the profile page to load dynamic content.")
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        for i in range(self._scroll_limit):
            logger.info(f"Scrolling iteration {i+1}/{self._scroll_limit}.")
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            # Wait for new content to load
            time.sleep(3)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                logger.info("Reached the bottom of the page.")
                break
            last_height = new_height

    def close(self) -> None:
        """Closes the Selenium driver."""
        logger.info("Closing the WebDriver.")
        self.driver.quit()

# Example usage:
if __name__ == "__main__":
    # Replace the following credentials with your actual LinkedIn username and password
    # WARNING: Hardcoding credentials is insecure. Use environment variables or secure storage in production.
    crawler = LinkedInCrawler(headless=False)  # Set headless=False for debugging
    test_user = {"id": str(uuid.uuid4()), "full_name": "Test User"}
    test_link = "https://www.linkedin.com/in/sundarpichai/"  # Example LinkedIn profile
    
    try:
        crawler.extract(link=test_link, user=test_user)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        crawler.close()