#!/usr/bin/env python

import codecs
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import keyboardlayout as kl
import keyboardlayout.pygame as klp
import librosa
# import pyrubberband as pyrb #look into this
import numpy
import pygame
import soundfile

ANCHOR_INDICATOR = " anchor"
ANCHOR_NOTE_REGEX = re.compile(r"\s[abcdefg]$")
DESCRIPTION = 'Use your computer keyboard as a "piano"'
DESCRIPTOR_32BIT = "FLOAT"
BITS_32BIT = 32
AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED = 0
SOUND_FADE_MILLISECONDS = 50
CYAN = (0, 255, 255, 255)
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)

AUDIO_ASSET_PREFIX = "audio_files/"
KEYBOARD_ASSET_PREFIX = "keyboards/"
CURRENT_WORKING_DIR = Path(__file__).parent.absolute()
ALLOWED_EVENTS = {pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT}

wav_file_path = "pianoputer/audio_files/bowl_c6.wav"
keyboard_path = "pianoputer/keyboards/qwerty_piano.txt"

def get_or_create_key_sounds(
    wav_path: str,
    sample_rate_hz: int,
    channels: int,
    tones: List[int],
) -> Generator[pygame.mixer.Sound, None, None]:

    sounds = []
    y, sr = librosa.load(wav_path, sr=sample_rate_hz, mono=channels == 1)
    print("Loading notes...")
    for tone in (tones):
        if channels == 1:
            sound = librosa.effects.pitch_shift(y, sr=sample_rate_hz, n_steps=tone)
        else:
            new_channels = [
                librosa.effects.pitch_shift(y[i], sr=sample_rate_hz, n_steps=tone)
                for i in range(channels)
            ]
            sound = numpy.ascontiguousarray(numpy.vstack(new_channels).T)
        sounds.append(sound)
    sounds = map(pygame.sndarray.make_sound, sounds)
    return sounds



BLACK_INDICES_C_SCALE = [1, 3, 6, 8, 10]
LETTER_KEYS_TO_INDEX = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

def __get_black_key_indices(key_name: str) -> set:
    letter_key_index = LETTER_KEYS_TO_INDEX[key_name]
    black_key_indices = set()
    for ind in BLACK_INDICES_C_SCALE:
        new_index = ind - letter_key_index
        if new_index < 0:
            new_index += 12
        black_key_indices.add(new_index)
    return black_key_indices

def get_keyboard_info(keyboard_file: str):
    with codecs.open(keyboard_file, encoding="utf-8") as k_file:
        lines = k_file.readlines()
    keys = []
    anchor_index = -1
    black_key_indices = set()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        match = ANCHOR_NOTE_REGEX.search(line)
        if match:
            anchor_index = i
            black_key_indices = __get_black_key_indices(line[-1])
            key = kl.Key(line[: match.start(0)])
        elif line.endswith(ANCHOR_INDICATOR):
            anchor_index = i
            key = kl.Key(line[: -len(ANCHOR_INDICATOR)])
        else:
            key = kl.Key(line)
        keys.append(key)
    if anchor_index == -1:
        raise ValueError(
            "Invalid keyboard file, one key must have an anchor note or the "
            "word anchor written next to it.\n"
            "For example 'm c OR m anchor'.\n"
            "That tells the program that the wav file will be used for key m, "
            "and all other keys will be pitch shifted higher or lower from "
            "that anchor. If an anchor note is used then keys are colored black "
            "and white like a piano. If the word anchor is used, then the "
            "highest key is white, and keys get darker as they descend in pitch."
        )
    tones = [i - anchor_index for i in range(len(keys))]
    color_to_key = defaultdict(list)
    if black_key_indices:
        key_color = (120, 120, 120, 255)
        key_txt_color = (50, 50, 50, 255)
    else:
        key_color = (65, 65, 65, 255)
        key_txt_color = (0, 0, 0, 255)
    for index, key in enumerate(keys):
        if index == anchor_index:
            color_to_key[CYAN].append(key)
            continue
        if black_key_indices:
            used_index = (index - anchor_index) % 12
            if used_index in black_key_indices:
                color_to_key[BLACK].append(key)
                continue
            color_to_key[WHITE].append(key)
            continue
        # anchor mode, keys go up in half steps and we do not color black keys
        # instead we color from grey low to white high
        rgb_val = 255 - (len(keys) - 1 - index) * 3
        color = (rgb_val, rgb_val, rgb_val, 255)
        color_to_key[color].append(key)

    return keys, tones, color_to_key, key_color, key_txt_color

def configure_pygame_audio_and_set_ui(
    framerate_hz: int,
    channels: int,
    keyboard_arg: str,
    color_to_key: Dict[str, List[kl.Key]],
    key_color: Tuple[int, int, int, int],
    key_txt_color: Tuple[int, int, int, int],
) -> Tuple[pygame.Surface, klp.KeyboardLayout]:
    # ui
    pygame.display.init()
    pygame.display.set_caption("pianoputer")

    # block events that we don't want, this must be after display.init
    pygame.event.set_blocked(None)
    pygame.event.set_allowed(list(ALLOWED_EVENTS))

    # fonts
    pygame.font.init()

    # audio
    pygame.mixer.init(
        framerate_hz,
        BITS_32BIT,
        channels,
        allowedchanges=AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED,
    )

    screen_width = 50
    screen_height = 50
    if "qwerty" in keyboard_arg:
        layout_name = kl.LayoutName.QWERTY
    elif "azerty" in keyboard_arg:
        layout_name = kl.LayoutName.AZERTY_LAPTOP
    else:
        ValueError("keyboard must have qwerty or azerty in its name")
    margin = 4
    key_size = 60
    overrides = {}
    for color_value, keys in color_to_key.items():
        override_color = color = pygame.Color(color_value)
        inverted_color = list(~override_color)
        other_val = 255
        if (
            abs(color_value[0] - inverted_color[0]) > abs(color_value[0] - other_val)
        ) or color_value == CYAN:
            override_txt_color = pygame.Color(inverted_color)
        else:
            # biases grey override keys to use white as txt_color
            override_txt_color = pygame.Color([other_val] * 3 + [255])
        override_key_info = kl.KeyInfo(
            margin=margin,
            color=override_color,
            txt_color=override_txt_color,
            txt_font=pygame.font.SysFont("Arial", key_size // 4),
            txt_padding=(key_size // 10, key_size // 10),
        )
        for key in keys:
            overrides[key.value] = override_key_info

    key_txt_color = pygame.Color(key_txt_color)
    keyboard_info = kl.KeyboardInfo(position=(0, 0), padding=2, color=key_txt_color)
    key_info = kl.KeyInfo(
        margin=margin,
        color=pygame.Color(key_color),
        txt_color=pygame.Color(key_txt_color),
        txt_font=pygame.font.SysFont("Arial", key_size // 4),
        txt_padding=(key_size // 6, key_size // 10),
    )
    letter_key_size = (key_size, key_size)  # width, height
    keyboard = klp.KeyboardLayout(
        layout_name, keyboard_info, letter_key_size, key_info, overrides
    )
    screen_width = keyboard.rect.width
    screen_height = keyboard.rect.height

    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill(pygame.Color("black"))
    if keyboard:
        keyboard.draw(screen)
    pygame.display.update()
    return screen, keyboard

def play_until_user_exits(
    keys: List[kl.Key],
    key_sounds: List[pygame.mixer.Sound],
    keyboard: klp.KeyboardLayout,
):
    sound_by_key = dict(zip(keys, key_sounds))
    playing = True

    while playing:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                playing = False
                break
            elif event.key == pygame.K_ESCAPE:
                playing = False
                break

            key = keyboard.get_key(event)
            if key is None:
                continue
            try:
                sound = sound_by_key[key]
            except KeyError:
                continue

            if event.type == pygame.KEYDOWN:
                sound.stop()
                sound.play(fade_ms=SOUND_FADE_MILLISECONDS)
            elif event.type == pygame.KEYUP:
                sound.fadeout(SOUND_FADE_MILLISECONDS)

    pygame.quit()
    print("Goodbye")

def get_audio_data(wav_path: str) -> Tuple:
    audio_data, framerate_hz = soundfile.read(wav_path)
    array_shape = audio_data.shape
    if len(array_shape) == 1:
        channels = 1
    else:
        channels = array_shape[1]
    return audio_data, framerate_hz, channels

def play_pianoputer():
    audio_data, framerate_hz, channels = get_audio_data(wav_file_path)
    results = get_keyboard_info(keyboard_path)
    keys, tones, color_to_key, key_color, key_txt_color = results
    key_sounds = get_or_create_key_sounds(
        wav_file_path, framerate_hz, channels, tones
    )

    _screen, keyboard = configure_pygame_audio_and_set_ui(
        framerate_hz, channels, keyboard_path, color_to_key, key_color, key_txt_color
    )
    print(
        "Ready for you to play!\n"
        "Press the keys on your keyboard. "
        "To exit presss ESC or close the pygame window"
    )
    play_until_user_exits(keys, key_sounds, keyboard)

if __name__ == "__main__":
    play_pianoputer()
