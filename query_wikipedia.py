import requests
from bs4 import BeautifulSoup

wikiurl = "https://ru.wikipedia.org/wiki/"

def get_reply(query):
    response = requests.get(wikiurl + query)
    print(response.status_code)
    soup = BeautifulSoup(response.content, 'html.parser')
    annotation = soup.find('div', {'id': "mw-content-text"}).find('p')
    return annotation.text
