#Разметка SSML стихотворения

#espeak, espeak-ng, google, amazon, ms, ...
import subprocess
import win32com.client
import io

speaker = win32com.client.Dispatch("SAPI.SpVoice")

path = '"c:\\Program Files\\eSpeak NG\\espeak-ng.exe" -m -v ru -f rhyme.ssml'

#result = subprocess.run(path, stdout=subprocess.PIPE)
#print(result.stdout.decode('utf-8'))

file = io.open('rhyme.ssml', mode="r", encoding="utf-8")
data = file.read()
speaker.speak(data, 264)
