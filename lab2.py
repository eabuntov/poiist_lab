#Разметка SSML стихотворения

#espeak, espeak-ng, google, amazon, ms, ...
import subprocess
import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")

for voice in speaker.GetVoices():
    print(voice.GetDescription())


path = '"c:\\Program Files\\eSpeak NG\\espeak-ng.exe" -m -v ru -f rhyme.ssml'

result = subprocess.run(path, stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))

#speaker.speak(phrase, 384)
