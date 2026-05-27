"""
debug_as24_item.py — Inspecte la structure d'une annonce AutoScout24
"""
import requests
from bs4 import BeautifulSoup
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.fr/",
}

url = "https://www.autoscout24.fr/lst/bmw/5er?sort=standard&desc=0&ustate=N,U&size=20&page=1&cy=F"
resp = requests.get(url, headers=HEADERS, timeout=15)
soup = BeautifulSoup(resp.text, "html.parser")
nd   = soup.find("script", id="__NEXT_DATA__")
data = json.loads(nd.string)
listings = data["props"]["pageProps"]["listings"]

print(f"Nombre d'annonces : {len(listings)}\n")

# Inspecte les 3 premières annonces en détail
for i, item in enumerate(listings[:3]):
    print(f"═══ Annonce {i+1} ═══")
    print(f"  id      : {item.get('id')}")
    print(f"  url     : {item.get('url')}")
    print(f"  price   : {json.dumps(item.get('price'), ensure_ascii=False)}")
    print(f"  vehicle : {json.dumps(item.get('vehicle'), ensure_ascii=False)}")
    print(f"  location: {json.dumps(item.get('location'), ensure_ascii=False)}")
    print(f"  vehicleDetails: {json.dumps(item.get('vehicleDetails'), ensure_ascii=False)[:300]}")
    print()