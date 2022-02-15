import time
import sys
import vlc

def play_sound(name):
    sound = vlc.MediaPlayer('sounds/' + name + '.mp3')
    sound.play()
    time.sleep(0.8)
    sound.stop()
    sound.release()

def pronounce(digit, order):
    play_sound(str(digit))
    if order == 6:
        play_sound('mlns')
    elif order == 3:
        play_sound('1000s')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Too few arguments!")
    elif sys.argv[1].isdigit():
        num = int(sys.argv[1])
        order = len(sys.argv[1])
        if 0 < num < 1000000000:
            print("Pronouncing {}".format(num))
            for n in range(order, -1, -1):
                digit = num // (10 ** n)
                num -= digit * 10 ** n
                if digit > 0:
                    print(digit)
                    pronounce(digit, n)
                    #time.sleep(0.5)
        else:
            print("Number out of range!")
    else:
        print("Not a number!")

