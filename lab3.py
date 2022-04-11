import argparse
import requests
import os
import queue
import sounddevice as sd
import vosk
import sys
import json
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from query_wikipedia import process_p, process_l
import win32com.client
import re
import matplotlib
matplotlib.use('TKAgg')

head = """<?xml version="1.0"?>
<speak  version="1.0" xml:lang="ru">\n"""

tail = "\n</speak>\n"

speaker = win32com.client.Dispatch("SAPI.SpVoice")

def say(str, text_mode, sp = speaker):
    text = re.sub(r"([1-3][0-9]{3})", "<say-as interpret-as=\"ordinal\">" + r"\1" + r"</say-as>", str)
    if text_mode:
        print(head + text + tail)
    else:
        sp.speak(head + text + tail, 264)


def listen(text_mode, stream):
    if text_mode:
        return input()
    else:
        with stream:
            data = q.get()
            if rec.AcceptWaveform(data):
                return json.loads(rec.Result())['text']
            else:
                return json.loads(rec.PartialResult())['partial']


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
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata)



def update_plot(frame):
    """This is called by matplotlib for each plot update.
    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.
    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

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
text_mode = False
parser.add_argument(
    '-t', '--text-mode', action='store_true',
    help='Use text mode without sound')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
args = parser.parse_args(remaining)

try:
    if args.model is None:
        args.model = "vosk-model-small-ru-0.22"
    if args.text_mode:
        text_mode = True
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
    plotdata = np.zeros((3*args.samplerate, 1))
    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)
    stream = sd.InputStream(
        device=args.device, channels=1,
        samplerate=args.samplerate, callback=callback)
    ani = FuncAnimation(fig, update_plot, interval=30, blit=True)
    with stream:
        plt.show()

    say("Привет!", text_mode)
    while True:  # Create a little chatbot
        query_str = ""
        ans = ""
        say("Что поискать?", text_mode)
        query_str = listen(text_mode, stream)
        if query_str == "выход":
            say("Спасибо, до свидания!", text_mode)
            break
        if len(query_str) > 0:
            say(f"Ищем {query_str}...", text_mode)
            try:
                url = f"https://ru.wikipedia.org/wiki/{query_str}"
                page = requests.get(url)
                say("Вот, что я нашла:", text_mode)
                if "<p>В Википедии <b>нет статьи</b> с таким названием." in page.text:
                    say("Страница не найдена. Попробуйте переформулировать запрос", text_mode)
                else:
                    text_info = BeautifulSoup(page.content, 'html.parser').find_all("p")
                    ans = process_p(text_info)
                    if ans:
                        for line in ans.splitlines(False):
                            say(line, text_mode)
                            say("Вас устраивает ответ?", text_mode)
                            ans = listen(text_mode, stream)
                            if ans.strip().lower() == "да":
                                say("Не сто'ит благодарности! Для завершения скажите \"выход\"", text_mode)
                                break
                    else:
                        links = BeautifulSoup(page.content, 'html.parser').find_all("li")
                        ans = process_l(links)
                        if ans:
                            for line in ans.splitlines(False):
                                say(line.split('/')[0], text_mode)
                                say("Вас устраивает ответ?", text_mode)
                                an = listen(text_mode, stream)
                                if an.strip().lower() == "да":
                                    say("Прочитать оригинальную статью?", text_mode)
                                    an = listen(text_mode)
                                    if an.strip().lower() == "да":
                                        url = f"https://ru.wikipedia.org/{line.split('/')[1]+'/'+line.split('/')[2]}"
                                        page = requests.get(url)
                                        text_info = BeautifulSoup(page.content, 'html.parser').find_all("p")
                                        ans = process_p(text_info)
                                        for line in ans.splitlines(False):
                                            say(line, text_mode)
                                            say("Вас устраивает ответ?", text_mode)
                                            ans = listen(text_mode)
                                            if ans.strip().lower() == "да":
                                                say("Не сто'ит благодарности! Для завершения скажите \"выход\"",
                                                    text_mode)
                                                break
                                    say("Для завершения скажите выход", text_mode)
                                    break
            except:
                break
        else:
            say("Пожалуйста повторите", text_mode)
except KeyboardInterrupt:
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
