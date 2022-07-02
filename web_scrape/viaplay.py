from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time

#/usr/bin/safaridriver
driver = webdriver.Safari()
#link to viaplay player
driver.get("https://viaplay.no/player/default/serier/californication/sesong-2/episode-11")

#<button id="onetrust-accept-btn-handler">Accept All Cookies</button>
button = driver.find_element(By.ID, "onetrust-accept-btn-handler")
button.click()

email = "teamgreen22@hotmail.com"
password = "JosteinFlo666"


#<input name="username" data-testhook="login-username" type="email" autocomplete="username" placeholder="E-post" value="">
mail_fill = driver.find_element(By.NAME, "username")
mail_fill.send_keys(email)
#<input name="password" data-testhook="login-password" type="password" autocomplete="current-password" placeholder="Passord" value="">
pass_fill = driver.find_element(By.NAME, "password")
pass_fill.send_keys(password)
#<input class="Buttons-primary-3n82B LoginForm-button-lZCQj" type="submit" data-testhook="login-button" value="Logg inn">
button = driver.find_element(By.NAME, "login-button")
time.sleep(10)
