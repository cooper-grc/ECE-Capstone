import pygame
import rtmidi
import os
import RPi.GPIO as GPIO
import threading

pygame.mixer.pre_init(44100, -16, 2, 2048)
pygame.mixer.init()
pygame.init()

resourceLock = threading.Lock()

input = 0

low = -23
high = 19

mid = 60

sounds =[]

for note in range(low,high+1):
    newsound = pygame.mixer.Sound("/path/to/audio_files" + str(note)+".wav")
    sounds.append(newsound)

def handle_midi(event, data=sounds):
    if(resourceLock.acquire(blocking=False)):
        num = event.msg[1]
        index = num - mid - low
        pygame.mixer.Sound.play(sounds[index])
    else:
        pass


thispid = os.getpid()
os.system("sudo renice -20 "+ thispid) #set priority

#midi initializations
midi_in = rtmidi.MidiIn()
midi_in.open_port(1)
midi_in.set_callback(handle_midi)

#placeholder for i2s input in the future
while True:
    if input:
        if(resourceLock.acquire(blocking=False)):
            print("lock acquired")
        else:
            continue
