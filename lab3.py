import argparse
import time

import requests
import os
import queue
import sounddevice as sd
import vosk
import sys
import json
from bs4 import BeautifulSoup
import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")

q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-m', '--model', type=str, metavar='MODEL_PATH',
    help='Path to the model')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
args = parser.parse_args(remaining)

try:
    if args.model is None:
        args.model = "vosk-model-small-ru-0.22"
    if not os.path.exists(args.model):
        print ("Please download a model for your language from https://alphacephei.com/vosk/models")
        print ("and unpack as 'model' in the current folder.")
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])

    model = vosk.Model(args.model)
    print('#' * 80)
    print('Press Ctrl+C to stop the recording')
    print('#' * 80)

    rec = vosk.KaldiRecognizer(model, args.samplerate)

    while True:  # Create a little chatbot
        query_str = ""
        ans = ""
        speaker.speak("Привет, хозяин! Что поискать?")
        #time.sleep(5)
        with sd.RawInputStream(samplerate=args.samplerate, blocksize=args.samplerate * 3, device=args.device,
                               dtype='int16',
                               channels=1, callback=callback):
            data = q.get()
            if rec.AcceptWaveform(data):
                query_str = json.loads(rec.Result())['text']
            else:
                query_str = json.loads(rec.PartialResult())['partial']

        speaker.speak(f"Ищем {query_str}...")
        if len(query_str) > 0:
            try:
                url = f"https://ru.wikipedia.org/wiki/{query_str}"
                page = requests.get(url)
                text_info = BeautifulSoup(page.content, 'html.parser').find_all("p")
                speaker.speak("Вот, что я нашла:")
                for i in text_info:
                   if i.text != "\n":
                        speaker.speak(i.text)
                        break
                speaker.speak("Вас устраивает ответ?")
                #time.sleep(5)
                with sd.RawInputStream(samplerate=args.samplerate, blocksize=args.samplerate * 3,
                                       device=args.device, dtype='int16',
                                       channels=1, callback=callback):
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        ans = json.loads(rec.Result())['text']
                    else:
                        ans = json.loads(rec.PartialResult())['partial']
                if ans.strip().lower() == "да":
                    speaker.speak("Я так и думала. Не сто'ит благодарности!")
                    break
                else:
                    speaker.speak("Простите, хозяин. Я ничего не нашла. Спросите еще раз!")
            except:
                break
except KeyboardInterrupt:
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
