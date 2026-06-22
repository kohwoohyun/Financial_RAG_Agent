import requests
import zipfile
import io
import os

api_key = os.getenv("DART_API_KEY")
url = "https://opendart.fss.or.kr/api/corpCode.xml"
params = {"crtfc_key": api_key}

response = requests.get(url, params=params)

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("data")

print("완료: data/CORPCODE.xml 생성됨")
