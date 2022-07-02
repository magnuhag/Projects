from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd

#/usr/bin/safaridriver
driver = webdriver.Safari()
#link to viaplay player
driver.get("https://viaplay.no/player/default/serier/californication/sesong-2/episode-11")

#<button id="onetrust-pc-btn-handler" class="cookie-setting-link">Cookies Settings</button>
button1 = driver.find_element(By.ID, "onetrust-pc-btn-handler")
button1.click()

#<button class="save-preference-btn-handler onetrust-close-btn-handler" tabindex="0">Confirm My Choices</button>
button2 = driver.find_element(By.NAME, "save-preference-btn-handler onetrust-close-btn-handler")

#<button class="skip-preliminaries-button">Hopp over</button>
