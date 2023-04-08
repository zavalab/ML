import numpy as np
import matplotlib.pyplot as plt
import selenium
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select



driver = webdriver.Chrome("PATH_to_chromedriver")
driver.set_page_load_timeout("20")
# Load webpage
driver.get("https://dnr.wi.gov/lakes/waterquality/")
time.sleep(2)

#/html/body/form/div[5]/div/div[1]/div[2]/ul/li[1]/a
#/html/body/form/div[5]/div/div[1]/div[3]/ul/li[35]/a

def get_csv(table_num):
    driver.find_element_by_xpath(f"/html/body/form/div[5]/div/div[1]/div[3]/table/tbody/tr[{table_num}]/td[5]/a").click()
                                   
    time.sleep(1.5)
    for (j, names) in enumerate(driver.find_elements_by_xpath("/html/body/form/div[5]/div/div[1]/a")):
        if names.text == "Data Download":
            lake_csv = names.get_attribute("href")
            driver.get(f"{lake_csv}")
            time.sleep(1.5)
    driver.back()
    time.sleep(1.5)


for k in range(69,72):
    classes = driver.find_elements_by_class_name("multiColList")
    classes[k].click()

    time.sleep(3)

    for j in range(26):
        driver.find_element_by_xpath(f"/html/body/form/div[5]/div/div[1]/div[2]/div/a[{j+1}]").click()
        time.sleep(1.5)


        if len(driver.find_elements_by_xpath("/html/body/form/div[5]/div/div[1]/div[3]/table/tbody/tr")) > 0:
            for t1 in range(1,len(driver.find_elements_by_xpath("/html/body/form/div[5]/div/div[1]/div[3]/table/tbody/tr"))):
                names = driver.find_element_by_xpath(f"/html/body/form/div[5]/div/div[1]/div[3]/table/tbody/tr[{t1+1}]/td[1]")
                if "Deep Hole" in (names.text) or "Max Depth" in (names.text) or "Deepest" in (names.text) or "Maximum Depth" in (names.text):
                    if k == 0 and j == 11 and t1 == 1:
                        print("Skipping that lake; it just throws an error :P ")
                    elif k == 3 and j == 3 and t1 == 1:
                        print("Skipping that one too!")
                    elif k == 6 and j == 17 and t1 == 4:
                        print("There's another one. :( ")
                    else:
                        get_csv(t1+1)
                        print("Done with: county = ", k, ",  letter = ", j, ",  lake = ", t1)

                        if "Deep" in driver.find_element_by_xpath("/html/body/form/div[5]/div/div[1]/h1").text or "Depth" in driver.find_element_by_xpath("/html/body/form/div[5]/div/div[1]/h1").text:
                            driver.back()
                            time.sleep(1.5)
    time.sleep(3)
    driver.get("https://dnr.wi.gov/lakes/waterquality/")
    time.sleep(3)
    

#Error /html/body/div[4]/div[2]/h3


#/html/body/form/div[5]/div/div[1]/div[2]/div/a[1]
#/html/body/form/div[5]/div/div[1]/div[2]/div/a[26]




