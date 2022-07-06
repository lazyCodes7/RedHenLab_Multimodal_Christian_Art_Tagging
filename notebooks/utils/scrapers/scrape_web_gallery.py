from bs4 import BeautifulSoup
import requests
def retrieve_description(row):
    soup = BeautifulSoup(requests.get(row).text)
    for desc in soup.findAll('p'):
        return desc.text

