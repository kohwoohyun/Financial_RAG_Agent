import xml.etree.ElementTree as ET

tree = ET.parse("data/CORPCODE.xml")
root = tree.getroot()

targets = ["삼성전자", "SK하이닉스", "현대자동차", "LG에너지솔루션", "NAVER"]

for company in root.findall("list"):
    name = company.findtext("corp_name")
    if name in targets:
        code = company.findtext("corp_code")
        print(f"{name}: {code}")
