import re

wikiurl = "https://ru.wikipedia.org/wiki/"

def process_p(paragraphs, links):
    for i in paragraphs:
        if "—" in i.text and len(i.text) > 50:
            return re.sub(r'\[[^)]*\]', '', re.sub(r'\([^)]*\)', '', i.text))
    ans = ""
    for i in links:
        if "—" in i.text and len(i.text) > 50 and "Wikipedia" not in i.text:
            ans += re.sub(r'\[[^)]*\]', '', re.sub(r'\([^)]*\)', '', i.text.replace(".", ".\n")))
    if len(ans) > 0:
        return ans
    else:
        return "Ответ не найден"


def process_l(items):
    res = ""
    try:
        for i in items:
            res += i.find("a")[0]['title'] + ' '
    except:
        print('oops')
    return res

