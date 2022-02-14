import time
import sys
import vlc


def pronounce(sound, start, duration):
    sound.play()
    sound.set_time(int(start*1000))
    time.sleep(duration)
    sound.stop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Too few arguments!")
    elif sys.argv[1].isdigit():
        num = int(sys.argv[1])
        order = len(sys.argv[1])
        if 0 < num < 1000000000:
            print("Pronouncing {}".format(num))
            for n in range(order, 0, -1):
                digit = num // (10 ** n)
                num -= digit * 10 ** n
                if digit > 0:
                    print(digit)
                    sound = vlc.MediaPlayer('numbers.mp3')
                    pronounce(sound, digit*0.7, 0.75)
            sound.release()
        else:
            print("Number out of range!")
    else:
        print("Not a number!")

