from selenium import webdriver
import os
import time
import urllib


path = 'C:\Program Files (x86)\chromedriver.exe'

#save_folder = "D:/Keep_babies_safe_kaggle/training_images/toys"
save_folder = "D:/Liveliness_test/hand_detection/Hand_images/"

driver = webdriver.Chrome(path)
#driver.get("https://www.google.com/")

#https://www.google.com/search?q=cat&tbm=isch&ved=2ahUKEwjpsvLht8DsAhWOhUsFHRayCxoQ2-cCegQIABAA&oq=cat&gs_lcp=CgNpbWcQAzIECCMQJzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIHCAAQsQMQQzIECAAQQzIHCAAQsQMQQzIECAAQQzIECAAQQzoFCAAQsQM6BwgjEOoCECdQmyZY2S9g_jFoAXAAeAKAAbkBiAGcBpIBAzAuNpgBAKABAaoBC2d3cy13aXotaW1nsAEKwAEB&sclient=img&ei=ymiNX-mlIY6LrtoPluSu0AE&bih=666&biw=1536

#driver.get("https://www.google.com/search?q=dog&tbm=isch&ved=2ahUKEwjpsvLht8DsAhWOhUsFHRayCxoQ2-cCegQIABAA&oq=cat&gs_lcp=CgNpbWcQAzIECCMQJzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIHCAAQsQMQQzIECAAQQzIHCAAQsQMQQzIECAAQQzIECAAQQzoFCAAQsQM6BwgjEOoCECdQmyZY2S9g_jFoAXAAeAKAAbkBiAGcBpIBAzAuNpgBAKABAaoBC2d3cy13aXotaW1nsAEKwAEB&sclient=img&ei=ymiNX-mlIY6LrtoPluSu0AE&bih=666&biw=1536")

url_prefix = "https://www.google.com/search?q="
url_postfix = "&tbm=isch&ved=2ahUKEwjpsvLht8DsAhWOhUsFHRayCxoQ2-cCegQIABAA&oq="

#url_prefix = "https://www.dreamstime.com/photos-images/"
#url_postfix = ".html"

#url_postfix_an = "&gs_lcp=CgNpbWcQAzIECCMQJzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIHCAAQsQMQQzIECAAQQzIHCAAQsQMQQzIECAAQQzIECAAQQzoFCAAQsQM6BwgjEOoCECdQmyZY2S9g_jFoAXAAeAKAAbkBiAGcBpIBAzAuNpgBAKABAaoBC2d3cy13aXotaW1nsAEKwAEB&sclient=img&ei=ymiNX-mlIY6LrtoPluSu0AE&bih=666&biw=1536""
#take_input = input("enter the name of thing you wanna search:  ")
#final_url = url_prefix + take_input + url_postfix




def main():
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    download_images()
    
def download_images():
    topic = input("What do you want to search for?:   ")
    n_images = int(input('How many images do you want?:   '))
    
    search_url = url_prefix+topic+url_postfix
    #print(search_url)
    
    path = r'C:\Program Files (x86)\chromedriver.exe'
    
    driver = webdriver.Chrome(path)
    driver.get(search_url)
    
    value = 0
    for i in range(50):
        driver.execute_script("scrollBy("+ str(value) +",+2000);")
        value += 2000
        time.sleep(1)
    
    elem1 = driver.find_element_by_id('islmp')
    sub = elem1.find_elements_by_tag_name('img')
    
    #count=0
    for j,i in enumerate(sub):
        if j < n_images:
            src = i.get_attribute('src')                         
            try:
                if src != None:
                    src  = str(src)
                    print(src)
                    
                    urllib.request.urlretrieve(src, os.path.join(save_folder, str(j)+'.jpg'))
                    #count = count + 1
                else:
                    raise TypeError
            except Exception as e:              #catches type error along with other errors
                print('fail with error {}'.format(e))
    
    driver.close()
    
if __name__ == "__main__":
    main()