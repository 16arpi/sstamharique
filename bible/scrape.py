from bs4 import BeautifulSoup
from pathlib import Path
import glob
import re
import pandas as pd

SENT_START = re.compile(r'^[0-9]+ ', flags=re.MULTILINE)

def root():
    print("urls")
    home = BeautifulSoup(Path("./bible/assets/home.html").read_text(), "html.parser")
    for a in home.select(".nav-tabs a"):
        href = "https://www.wordproject.org/bibles/am/" + a["href"]
        print(href)

def sub():
    df = pd.read_csv("./bible/urls/report_pages.csv")
    paths = df["path"].apply(lambda a: f"./bible/assets/sub/{a}").to_list()
    soups = [BeautifulSoup(Path(p).read_text(), "html.parser") for p in paths]
    urls = df["resolved_url"].to_list()
    print("urls")
    for url, soup in zip(urls, soups):
        for a in soup.select("a.chap"):
            file = a["href"]
            end = url.replace("1.htm", file)
            print(end)

def extract():
    df = pd.read_csv("./bible/urls/report.csv")
    paths = df["path"].apply(lambda a: f"./bible/assets/sub/{a}").to_list()
    ids = df["path"].apply(lambda a: a.replace('.html', ''))
    soups = [BeautifulSoup(Path(p).read_text(), "html.parser") for p in paths]
    urls = df["resolved_url"].to_list()

    mp3s = []
    for id, soup in zip(ids, soups):
        # mp3
        a = soup.select_one("a.download")
        href = a["href"]
        mp3s.append(href)

        # verses
        body = soup.select_one("#textBody")
        #print(SENT_START.sub('', body.text))
        with open(f"./bible/transcrib/{id}.txt", 'w') as export:
            export.write(SENT_START.sub('', body.text))

    rt = pd.DataFrame({
        "id": ids,
        "mp3": mp3s,
    })

    rt.to_csv("./bible/urls/mp3s.csv", index=None)

extract()
