import time
import pytest
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

@pytest.fixture(scope="module")
def driver():
    # Set up Chrome driver in headless mode
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    yield driver
    driver.quit()

def test_streamlit_home(driver):
    # Navigate to the Streamlit app; adjust the URL if needed
    driver.get("http://localhost:8501")
    # Allow some time for the page to load
    time.sleep(5)
    # Check if the page contains the expected title text
    assert "RAG-Power Survey Analysis" in driver.page_source