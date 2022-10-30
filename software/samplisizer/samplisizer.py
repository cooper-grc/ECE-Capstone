# This is an iteration on the current version of our code 
# I'm working on maping the keys on the midi keyboard into sounds while preserving the previous structure
# of the code as much as possible
# It is messy and when finialized will be cleaner and commented

#!/usr/bin/env python

import argparse
import codecs
import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import keyboardlayout as kl
import keyboardlayout.pygame as klp
import librosa
import numpy
import pygame as pg
import soundfile

import sys

import pygame.midi

ANCHOR_INDICATOR = " anchor"
ANCHOR_NOTE_REGEX = re.compile(r"\s[abcdefg]$")
DESCRIPTION = 'Use your computer keyboard as a "piano"'
DESCRIPTOR_32BIT = "FLOAT"
BITS_32BIT = 32
AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED = 0
SOUND_FADE_MILLISECONDS = 200
CYAN = (0, 255, 255, 255)
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)

AUDIO_ASSET_PREFIX = "audio_files/"
KEYBOARD_ASSET_PREFIX = "keyboards/"
CURRENT_WORKING_DIR = Path(__file__).parent.absolute()
ALLOWED_EVENTS = {pg.KEYDOWN, pg.KEYUP, pg.QUIT, pg.MIDIIN}

def configure_pygame_audio(
    framerate_hz: int,
    channels: int,
    # keyboard_arg: str,
    # color_to_key: Dict[str, List[kl.Key]],
    # key_color: Tuple[int, int, int, int],
    # key_txt_color: Tuple[int, int, int, int],
) -> Tuple[pg.Surface, klp.KeyboardLayout]:
    # ui
    # pg.display.init()
    # pg.display.set_caption("pianoputer")

    # block events that we don't want, this must be after display.init
    pg.event.set_blocked(None)
    pg.event.set_allowed(list(ALLOWED_EVENTS))

    # fonts
    pg.font.init()

    # audio
    pg.mixer.init(
        framerate_hz,
        BITS_32BIT,
        channels,
        allowedchanges=AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED,
    )

    screen_width = 50
    screen_height = 50
    # if "qwerty" in keyboard_arg:
    #     layout_name = kl.LayoutName.QWERTY
    # elif "azerty" in keyboard_arg:
    #     layout_name = kl.LayoutName.AZERTY_LAPTOP
    # else:
    #     ValueError("keyboard must have qwerty or azerty in its name")
    # margin = 4
    # key_size = 60
    # overrides = {}
    # for color_value, keys in color_to_key.items():
    #     override_color = color = pg.Color(color_value)
    #     inverted_color = list(~override_color)
    #     other_val = 255
    #     if (
    #         abs(color_value[0] - inverted_color[0]) > abs(color_value[0] - other_val)
    #     ) or color_value == CYAN:
    #         override_txt_color = pg.Color(inverted_color)
    #     else:
    #         # biases grey override keys to use white as txt_color
    #         override_txt_color = pg.Color([other_val] * 3 + [255])
    #     override_key_info = kl.KeyInfo(
    #         margin=margin,
    #         color=override_color,
    #         txt_color=override_txt_color,
    #         txt_font=pg.font.SysFont("Arial", key_size // 4),
    #         txt_padding=(key_size // 10, key_size // 10),
    #     )
    #     for key in keys:
    #         overrides[key.value] = override_key_info

    # key_txt_color = pg.Color(key_txt_color)
    # keyboard_info = kl.KeyboardInfo(position=(0, 0), padding=2, color=key_txt_color)
    # key_info = kl.KeyInfo(
    #     margin=margin,
    #     color=pg.Color(key_color),
    #     txt_color=pg.Color(key_txt_color),
    #     txt_font=pg.font.SysFont("Arial", key_size // 4),
    #     txt_padding=(key_size // 6, key_size // 10),
    # )
    # letter_key_size = (key_size, key_size)  # width, height
    # keyboard = klp.KeyboardLayout(
    #     layout_name, keyboard_info, letter_key_size, key_info, overrides
    # )
    # screen_width = keyboard.rect.width
    # screen_height = keyboard.rect.height

    screen = pg.display.set_mode((screen_width, screen_height))
    screen.fill(pg.Color("black"))
    # if keyboard:
    #     keyboard.draw(screen)
    pg.display.update()
    return screen #, keyboard

def get_or_create_key_sounds(
    wav_path: str,
    sample_rate_hz: int,
    channels: int,
    tones: List[int],
    clear_cache: bool,
    keys: List[str],
) -> Generator[pygame.mixer.Sound, None, None]:
    sounds = []
    y, sr = librosa.load(wav_path, sr=sample_rate_hz, mono=channels == 1)
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    folder_containing_wav = Path(wav_path).parent.absolute()
    cache_folder_path = Path(folder_containing_wav, file_name)
    if clear_cache and cache_folder_path.exists():
        shutil.rmtree(cache_folder_path)
    if not cache_folder_path.exists():
        print("Generating samples for each key")
        os.mkdir(cache_folder_path)
    for i, tone in enumerate(tones):
        cached_path = Path(cache_folder_path, "{}.wav".format(tone))
        if Path(cached_path).exists():
            print("Loading note {} out of {} for {}".format(i + 1, len(tones), keys[i]))
            sound, sr = librosa.load(cached_path, sr=sample_rate_hz, mono=channels == 1)
            if channels > 1:
                # the shape must be [length, 2]
                sound = numpy.transpose(sound)
        else:
            print(
                "Transposing note {} out of {} for {}".format(
                    i + 1, len(tones), keys[i]
                )
            )
            if channels == 1:
                sound = librosa.effects.pitch_shift(y, sr, n_steps=tone)
            else:
                new_channels = [
                    librosa.effects.pitch_shift(y[i], sr, n_steps=tone)
                    for i in range(channels)
                ]
                sound = numpy.ascontiguousarray(numpy.vstack(new_channels).T)
            soundfile.write(cached_path, sound, sample_rate_hz, DESCRIPTOR_32BIT)
        sounds.append(sound)
    sounds = map(pygame.sndarray.make_sound, sounds)
    return sounds

def get_keyboard_info():
    keys = []
    anchor_index = 12

    for i in range(25):
        keys.append(i)

    tones = [i - anchor_index for i in range(len(keys))]

    return keys, tones

def play_until_user_exits(
    # keys: List[kl.Key],
    # key_sounds: List[pg.mixer.Sound],
    # keyboard: klp.KeyboardLayout,
):
    # sound_by_key = dict(zip(keys, key_sounds))
    # From https://stackoverflow.com/questions/64818410/pg-read-midi-input
    # pg.fastevent.init()
    event_get = pg.event.get
    event_post = pg.event.post

    pg.midi.init()

    _print_device_info()

    device_id = 3

    if device_id is None:
        input_id = pg.midi.get_default_input_id()
    else:
        input_id = device_id

    print("using input_id :%s:" % input_id)
    i = pg.midi.Input(input_id)

    pg.display.set_mode((1, 1))

    going = True
    key = None
    while going:
        events = event_get()
        for e in events:
            if e.type in [pg.QUIT]:
                going = False
            if e.type in [pg.KEYDOWN]:

                key = keyboard.get_key(e)
                print(key)
                #going = False
            if e.type in [pg.midi.MIDIIN]:
               if e.__dict__.get('data1') != 0:
                    try:
                        print("midi " + str(key))
                        # sound = sound_by_key[key]
                        sound.stop()
                        sound.play(fade_ms=SOUND_FADE_MILLISECONDS)
                        sound.fadeout(SOUND_FADE_MILLISECONDS)
                    except KeyError:
                        continue



        if i.poll():
            midi_events = i.read(10)
            # convert them into pygame events.
            midi_evs = pg.midi.midis2events(midi_events, i.device_id)

            for m_e in midi_evs:
                event_post(m_e)

    del i
    pg.midi.quit()

    pg.quit()
    print("Goodbye")

def get_audio_data(wav_path: str) -> Tuple:
    audio_data, framerate_hz = soundfile.read(wav_path)
    array_shape = audio_data.shape
    if len(array_shape) == 1:
        channels = 1
    else:
        channels = array_shape[1]
    return audio_data, framerate_hz, channels   


def play_samplisizer(args: Optional[List[str]] = None):
    parser = get_parser()
    wav_path, keyboard_path, clear_cache = process_args(parser, args)
    audio_data, framerate_hz, channels = get_audio_data(wav_path)
    results = get_keyboard_info(keyboard_path)
    keys, tones, color_to_key, key_color, key_txt_color = results
    key_sounds = get_or_create_key_sounds(
        wav_path, framerate_hz, channels, tones, clear_cache, keys
    )

    _screen, keyboard = configure_pygame_audio(
        framerate_hz, channels, keyboard_path, color_to_key, key_color, key_txt_color
    )
    print(
        "Ready for you to play!\n"
        "Press the keys on your keyboard. "
        "To exit presss ESC or close the pygame window"
    )
    play_until_user_exits(keys, key_sounds, keyboard)

def print_device_info():
    pg.midi.init()
    _print_device_info()
    pg.midi.quit()


def _print_device_info():
    for i in range(pg.midi.get_count()):
        r = pg.midi.get_device_info(i)
        (interf, name, input, output, opened) = r

        in_out = ""
        if input:
            in_out = "(input)"
        if output:
            in_out = "(output)"

        print(
            "%2i: interface :%s:, name :%s:, opened :%s:  %s"
            % (i, interf, name, opened, in_out)
        )


def input_main(device_id=None):
    # pygame.init()
    pg.fastevent.init()
    event_get = pg.fastevent.get
    event_post = pg.fastevent.post

    pg.midi.init()

    _print_device_info()

    if device_id is None:
        input_id = pg.midi.get_default_input_id()
    else:
        input_id = device_id

    print("using input_id :%s:" % input_id)
    i = pg.midi.Input(input_id)

    pg.display.set_mode((1, 1))

    going = True
    while going:
        events = event_get()
        for e in events:
            if e.type in [pg.QUIT]:
                going = False
            if e.type in [pg.KEYDOWN]:
                going = False
            if e.type in [pg.midi.MIDIIN]:
                print(e)

        if i.poll():
            midi_events = i.read(10)
            # convert them into pygame events.
            midi_evs = pg.midi.midis2events(midi_events, i.device_id)

            for m_e in midi_evs:
                event_post(m_e)

    del i
    pg.midi.quit()


if __name__ == "__main__":
    play_samplisizer()