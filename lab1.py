import time
import sys
import vlc


def play_sound(name):
    if name == '0':
        return
    sound = vlc.MediaPlayer('sounds/' + name + '.mp3')
    sound.play()
    time.sleep(0.8)
    sound.stop()
    sound.release()


def pronounce(digit, order, next_digit, carry_flag, thouthands):
    two_digits = carry_flag
    if order == 6:
        if not two_digits:
            play_sound(str(digit))
            two_digits = False
        if not two_digits and digit == 1:
            play_sound('mln')
        elif 2 <= digit <= 4 and not two_digits:
            play_sound('mln1')
            two_digits = False
        else:
            play_sound('mlns')
            two_digits = False
    elif order == 3:
        if not two_digits and digit > 1:
            play_sound(str(digit))
            two_digits = False
        if digit == 1 and not two_digits:
            play_sound('odna')
            play_sound('1000')
        elif 2 <= digit <= 4 and not two_digits:
            play_sound('1000s')
        elif thouthands:
            play_sound('1000s1')
            two_digits = False
    elif order == 8 or order == 5 or order == 2:
        play_sound(str(digit * 100))
        two_digits = False
    elif order == 7 or order == 4 or order == 1:
        if digit == 1:
            two_digits = True
            play_sound(str(digit*10 + next_digit))
        else:
            play_sound(str(digit * 10))
            two_digits = False
    else:
        if not two_digits:
            play_sound(str(digit))
            two_digits = False
    return two_digits


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Too few arguments!")
    elif sys.argv[1].isdigit():
        num = int(sys.argv[1])
        order = len(sys.argv[1])
        if 0 < num < 1000000000:
            print("Pronouncing {}".format(num))
            thousands = (num - (num // 1000000) * 1000000) // 1000 > 0 #there are some thousands
            two_digits = False
            for n in range(order-1, -1, -1):
                digit = num // (10 ** n)
                num -= digit * 10 ** n
                #if digit > 0:
                print(digit)
                two_digits = pronounce(digit, n, num // 10 ** (n-1), two_digits, thousands)
                #time.sleep(0.5)
        else:
            print("Number out of range!")
    else:
        print("Not a number!")

