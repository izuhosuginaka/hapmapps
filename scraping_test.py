import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementClickInterceptedException

from selenium.webdriver.chrome import service as fs

from selenium.webdriver.common.by import By

QUERY = 'ネコ'  # 検索ワード

DRIVER_PATH = './chromedriver'

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
# options.add_argument('--start-maximized')
options.add_argument('--start-fullscreen')
options.add_argument('--disable-plugins')
options.add_argument('--disable-extensions')

# print(options)

chrome_service = fs.Service(executable_path=DRIVER_PATH)
driver = webdriver.Chrome(service=chrome_service, options=options)
driver.set_window_size('1200', '1000')

# Google画像検索ページを取得
url = f'https://www.google.com/search?q={QUERY}&tbm=isch'
driver.get(url)

#tmb_elems = driver.find_element(by=By.CSS_SELECTOR, value='#islmp img')
tmb_elems = driver.find_elements_by_css_selector('#islmp img')

imgframe_elem = driver.find_element_by_id('islsp')

print("tmb_elems")
print(tmb_elems)

# サムネイル画像の数を知りたい場合
tmb_alts = [tmb.get_attribute('alt') for tmb in tmb_elems]
count = len(tmb_alts) - tmb_alts.count('')

print(count)



RETRY_NUM = 3    # リトライ回数

EXCLUSION_URL = 'https://lh3.googleusercontent.com/'  # 除外対象url

HTTP_HEADERS = {'User-Agent': driver.execute_script('return navigator.userAgent;')}
print(HTTP_HEADERS)

k = 0
# サムネイル画像だけ処理したい場合
for tmb_elem, tmb_alt in zip(tmb_elems, tmb_alts):   
    

    print ("tmb_alt", tmb_alt)
    if tmb_alt == '':
        continue
    #print(tmb_elem)
    
    
    for i in range(RETRY_NUM):
        try:
            print(tmb_elem.is_displayed())
            tmb_elem.click()
            driver.save_screenshot(str(k) + '.png')
            k += 1
            
            alt = tmb_alt.replace("'", "\\'")
            try:
                img_elem = imgframe_elem.find_element_by_css_selector(f'img[alt=\'{alt}\']')
                print ("img_elem", img_elem)
                
                tmb_url = tmb_elem.get_attribute('src')  # サムネイル画像のsrc属性値
                url = img_elem.get_attribute('src')
                if EXCLUSION_URL in url:
                    url = ''
                    break
                elif url == tmb_url:  # src属性値が遷移するまでリトライ
                    print ("url == tmb_url")
                    #time.sleep(1)
                    #url = ''
                else:
                    break
                print ("url", url)
                
                path = "image" + str(k) + ".png"
                
                if url.startswith('https'):
                    try:
                       r = requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=10)
                       r.raise_for_status()
                       with open(path, 'wb') as f:
                           f.write(r.content)
                    except requests.exceptions.SSLError:
                       print('***** SSL エラー')
                       break
                
            except NoSuchElementException:
                print ("NoSuchElementException")
                continue
        
            
        
        except ElementClickInterceptedException:
            print("except!")
            driver.execute_script('arguments[0].scrollIntoView(true);', tmb_elem)
            time.sleep(1)
        else:
            break
    else:
        print("continue!")
        continue

#driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')

