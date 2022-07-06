import pandas as pd
from bs4 import BeautifulSoup
import requests
def scrape_pages(url):
    links = []
    image_links = []
    idx = 0
    print(url)
    page_available = True
    while(page_available):
        idx+=1
        resp = requests.get(url.format(idx))
        if(resp.status_code>400):
            page_available = False
            break
        else:
            soup = BeautifulSoup(resp.text)
            for link in soup.find_all("a", class_="m-listing__link"):
                links.append(link.get('href'))
                if(link.img == None):
                    image_links.append(None)
                else:
                    ilink = link.img.get('data-pin-media')
                    image_links.append(ilink)
    return links, image_links

def scrape_images():
    scraping_links = [
      "https://www.artic.edu/collection?q=saint&artwork_type_id=Painting&is_public_domain=1&page={}",
      "https://www.artic.edu/collection?q=saint&is_public_domain=1&artwork_type_id=Drawing and Watercolor&page={}",
      "https://www.artic.edu/collection?q=saint&is_public_domain=1&artwork_type_id=Print&page={}"
    ]
    image_links = []
    links = []
    for url in scraping_links:
        link_res, image_res = scrape_pages(url)
        image_links+=image_res
        links+=link_res

    return links, image_links

def scrape_metadata(df):
    for i in range(len(df)):
      response = requests.get(df.iloc[i]['link'])
      soup = BeautifulSoup(response.text)
      for metadata in soup.find_all('dl', class_ = "deflist o-blocks__block "):
        nsoup = BeautifulSoup(metadata.text)
        categories = nsoup.p.text.split("\n")
        categories = list(filter(None, categories))
        for x in range(0, 10, 2):
          df.at[i, categories[x]] = categories[x+1]
      for description in soup.find_all('div', class_ = "o-blocks"):
        df.at[i, 'description'] = description.p.text
        break

