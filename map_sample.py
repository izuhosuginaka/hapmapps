import sys
import geocoder
import json

texts = ["旭川", "函館", "小樽", "苫小牧", "帯広", "釧路", "網走", "稚内"]

def geo(name):
    Localname = name
    g = geocoder.google(Localname)
    return g.lat,g.lng

json_data = []
for text in texts:
    lat, lng = geo(text)
    print (text, lat, lng) 
    
    json_data.append({
                    "name": text,
                    "lat": lat,
                    "lng": lng
                    })

with open('data.json', 'w') as outfile:
    json.dump(json_data, outfile, indent=4, sort_keys=True)

sys.exit()


# -*- coding: utf-8 -*-
from lxml.html import parse
from urllib2 import urlopen
import geocoder
import json

def main():
    parsed = parse(urlopen('http://www.pref.kanagawa.jp/cnt/f1029/p70915.html'))
    doc = parsed.getroot()
    blog_node = doc.xpath('//div[@class="detail_free"]')[0]

    json_data = []
    for a in blog_node.xpath('descendant::a'):
        href = a.get("href")
        if href:
            lat,lng = geo(a.text)
            if lat: #緯度経度が取得できたもののみ
                json_data.append({
                                "name": a.text,
                                "lat": lat,
                                "lng": lng
                                })

    with open('data.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4, sort_keys=True)

def geo(name):
    Localname = name
    g = geocoder.google(Localname)
    return g.lat,g.lng

if __name__ == '__main__':
    main()

