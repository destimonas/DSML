import requests
def simple_scraper(url):
    response = requests.get(url)
    if response.status_code == 200:
        print("content :")
        print(response.text)
    else:
        print("Failed to fetch page.Status code:", response.status_code)


url_to_scraper = "http://ajce.in"
simple_scraper(url_to_scraper)
