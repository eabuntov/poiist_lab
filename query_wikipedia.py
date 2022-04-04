import re

wikiurl = "https://ru.wikipedia.org/wiki/"

def process_p(paragraphs):
    for i in paragraphs:
        if "—" in i.text and len(i.text) > 50:
            return re.sub(r'\[[^)]*\]', '', re.sub(r'\([^)]*\)', '', i.text))
    return None


def process_l(links):
    ans = ""
    for i in links:
        if "—" in i.text and len(i.text) > 50 and "Wikipedia" not in i.text:
            ans += re.sub(r'\[[^)]*\]', '', re.sub(r'\([^)]*\)', '', i.text))
            ans += i.find('a')['href'] + '\n'
    if len(ans) > 0:
        return ans
    else:
        return None

