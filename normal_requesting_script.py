import urllib.request
import numpy as np
import requests


url = 'localhost:5000/post'
myobj = {
    'proof': 'High-five',
    'test': 'https://imagizer.imageshack.com/img923/2184/DVfH20.jpg'
}

x = requests.post(url, json = myobj)

print(x.text)