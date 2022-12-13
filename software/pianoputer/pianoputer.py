#!/usr/bin/env python
from scipy.io.wavfile import read
import argparse
import codecs
import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import pygame
import pygame.midi
import keyboardlayout as kl
import keyboardlayout.pygame as klp
import rtmidi

import librosa
import numpy
import soundfile

import pyaudio
import wave
import webrtcvad

import RPi.GPIO as GPIO
from time import sleep

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
ALLOWED_EVENTS = {pygame.KEYDOWN, pygame.KEYUP, pygame.QUIT, pygame.MIDIIN}

# GPIO defines
btn_pin = 11
channel = 11
g_pin = 3
y_pin = 5
r_pin = 8


def get_parser() -> argparse.ArgumentParser:
    """Generate and return parser - unused in current implementation"""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    default_wav_file = "/home/okcpe/ece-capstone/software/pianoputer/piano_c4.wav"
    parser.add_argument(
        "--wav",
        "-w",
        metavar="FILE",
        type=str,
        default=default_wav_file,
        help="WAV file (default: {})".format(default_wav_file),
    )
    default_keyboard_file = "keyboards/qwerty_piano.txt"
    parser.add_argument(
        "--keyboard",
        "-k",
        metavar="FILE",
        type=str,
        default=default_keyboard_file,
        help="keyboard file (default: {})".format(default_keyboard_file),
    )
    parser.add_argument(
        "--clear-cache",
        "-c",
        default=False,
        action="store_true",
        help="deletes stored transposed audio files and recalculates them",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose mode")

    return parser


def get_or_create_key_sounds(
    wav_path: str,
    sample_rate_hz: int,
    channels: int,
    tones: List[int],
    clear_cache: bool,
    keys: List[str],
) -> Generator[pygame.mixer.Sound, None, None]:
    sounds = []
    """Pitch shifting the sounds for each key.

    Keyword arguments:
    wav_path -- the path to the sound file
    sample_rate_hz -- the freq of the audio
    channels -- number of audio channels
    tones -- list of the notes numbers
    clear_cache -- bool to clear cache
    keys -- list of the piano keys
    Return variables:
    sounds -- map of pygame sounds
    """

    """LIBROSA LOAD:
    Load an audio file as a floating point time series.
    Audio will be automatically resampled to the given rate.
    Parameters:
        wav_path -- path to input file
        sr -- target sampling rate
        mono -- boolean to make single channel
    Returns: 
        y -- audio time series
        sr -- sampling rate
    """
    y, sr = librosa.load(wav_path, sr=sample_rate_hz, mono=channels == 1)
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    folder_containing_wav = Path(wav_path).parent.absolute()
    cache_folder_path = Path(folder_containing_wav, file_name)
    if clear_cache and cache_folder_path.exists():
        shutil.rmtree(cache_folder_path)
    if not cache_folder_path.exists():
        print("Generating samples for each key")
        os.mkdir(cache_folder_path)
    
    # Generate audio samples for each key:
    for i, tone in enumerate(tones):
        cached_path = Path(cache_folder_path, "{}.wav".format(tone))
        if Path(cached_path).exists():
            print("Loading note {} out of {} for {}".format(i + 1, len(tones), keys[i]))
            # sound -- audio time series, sr -- sampling rate
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

            """LIBROSA PITCH SHIFT:
            Shift the pitch of a waveform by n_steps steps.
            A step is equal to a semitone if bins_per_octave is set to 12.
            Parameters:
                y -- audio time series. Multi-channel is supported.
                sr -- audio sampling rate of y
                n_steps=tone -- how many (fractional) steps to shift y
            Returns:
                sound -- The pitch-shifted audio time-series
            """
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


BLACK_INDICES_C_SCALE = [1, 3, 6, 8, 10]
LETTER_KEYS_TO_INDEX = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def __get_black_key_indices(key_name: str) -> set:
    """Calculate and return the indices of black keys"""
    letter_key_index = LETTER_KEYS_TO_INDEX[key_name]
    black_key_indices = set()
    for ind in BLACK_INDICES_C_SCALE:
        new_index = ind - letter_key_index
        if new_index < 0:
            new_index += 12
        black_key_indices.add(new_index)
    return black_key_indices


def get_keyboard_info(keyboard_file: str):
    """Generate keyboard info
    Returns:
        keys -- list of keys
        tones -- list of tones
        color_to_key -- dicts of colors mapped to keys
        key_color -- tuple of key color
        key_txt_color -- tuple of key text color
    """
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


def get_audio_data(wav_path: str) -> Tuple:
    """Get the data from the wave file.

    Keyword arguments:
    wav_path -- the path to the audio wave file
    Return variables:
    framerate_hz -- the frequency in hertz of the sound file
    channels -- the number of channels from sound
    """
    audio_data, framerate_hz = soundfile.read(wav_path)
    array_shape = audio_data.shape
    if len(array_shape) == 1:
        channels = 1
    else:
        channels = array_shape[1]
    return audio_data, framerate_hz, channels


def process_args(parser: argparse.ArgumentParser, args: Optional[List]) -> Tuple:
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Enable warnings from scipy if requested
    if not args.verbose:
        warnings.simplefilter("ignore")

    wav_path = args.wav
    if wav_path.startswith(AUDIO_ASSET_PREFIX):
        wav_path = os.path.join(CURRENT_WORKING_DIR, wav_path)

    keyboard_path = args.keyboard
    if keyboard_path.startswith(KEYBOARD_ASSET_PREFIX):
        keyboard_path = os.path.join(CURRENT_WORKING_DIR, keyboard_path)
    return wav_path, keyboard_path, args.clear_cache


def record_sound():
    """
    Records from mic for 3 seconds and trims dead-space at start of clip then saves to wav file
    """
    form_1 = pyaudio.paInt16 # 16-bit resolution
    chans = 1 # 1 channel
    samp_rate = 44100 # 44.1kHz sampling rate
    chunk = 4096 # 2^12 samples for buffer
    record_secs = 3 # seconds to record
    dev_index = 1 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = '/home/okcpe/ece-capstone/software/pianoputer/piano_c4.wav' # name of .wav file

    ready_to_record()
    audio = pyaudio.PyAudio() # create pyaudio instantiation

    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=chunk)

    print("recording")
    recording()
    frames = []

    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)

    processing()
    print("finished recording")

    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()



    # save the audio frames as .wav file
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    # Trim the audio
    start = 0
    _, ints = read(wav_output_filename)
    print(ints)

    for i in ints:
         start +=1
         if abs(i) > 300:
            start = start//4096
            break
    trimmed = frames[start:]
    print(start)
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(trimmed))
    wavefile.close()


def setup_gpio():
    """
    Setup GPIO pins for LED outputs and button input
    """
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(channel, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(g_pin, GPIO.OUT)
    GPIO.setup(y_pin, GPIO.OUT)
    GPIO.setup(r_pin, GPIO.OUT)

    GPIO.output(g_pin, GPIO.LOW)
    GPIO.output(y_pin, GPIO.LOW)
    GPIO.output(r_pin, GPIO.LOW)


def recording():
    """
    Turn on red LED and turn off other LEDs to indicate program is in recording state
    """
    GPIO.output(g_pin, GPIO.LOW)
    GPIO.output(y_pin, GPIO.LOW)
    GPIO.output(r_pin, GPIO.HIGH)


def processing():
    """
    Turn on yellow LED and turn off other LEDs to indicate program is in processing state
    """
    GPIO.output(g_pin, GPIO.LOW)
    GPIO.output(y_pin, GPIO.HIGH)
    GPIO.output(r_pin, GPIO.LOW)


def ready():
    """
    Turn on green LED and turn off other LEDs to indicate program is in playable state
    """
    print("Playable")
    GPIO.output(g_pin, GPIO.HIGH)
    GPIO.output(y_pin, GPIO.LOW)
    GPIO.output(r_pin, GPIO.LOW)


def ready_to_record():
    """
    Blink red LED to warn user the program is about to enter the recording state
    """
    GPIO.output(g_pin, GPIO.LOW)
    GPIO.output(y_pin, GPIO.LOW)
    GPIO.output(r_pin, GPIO.HIGH)
    sleep(0.25)
    GPIO.output(r_pin, GPIO.LOW)
    sleep(0.25)
    GPIO.output(r_pin, GPIO.HIGH)
    sleep(0.25)
    GPIO.output(r_pin, GPIO.LOW)
    sleep(0.25)
    GPIO.output(r_pin, GPIO.HIGH)
    sleep(0.25)
    GPIO.output(r_pin, GPIO.LOW)
    sleep(0.22)


def error_indicator():
    """
    Turn on all of the LEDs to indicate the program encountered an error
    """
    GPIO.output(g_pin, GPIO.HIGH)
    GPIO.output(y_pin, GPIO.HIGH)
    GPIO.output(r_pin, GPIO.HIGH)


def play_pianoputer(args: Optional[List[str]] = None):
    """Organize the data and trigger the playing of the samplisizer.

    Keyword arguments:
        args -- list of arguments 
    """
    # Setup GPIO
    setup_gpio()
    processing()

    # Initialize pygame and midi
    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(1)
    midi_in.set_callback(handle_midi_input)

    #lower and higher range of wav files
    high= 19
    low = -23

    # Information variables from parser
    parser = get_parser()
    wav_path, keyboard_path, clear_cache = process_args(parser, args)

    while True:
        sounds =[]
        for i in range(low,high):
            try:
                newsound = pygame.mixer.Sound("/home/okcpe/ece-capstone/software/pianoputer/piano_c4/" + str(i)+".wav")
                sounds.append(newsound)
            except FileNotFoundError:
                pass

        # Pull audio data from wave file
        audio_data, framerate_hz, channels = get_audio_data(wav_path)
        # Pull keyboard info from path
        results = get_keyboard_info(keyboard_path)
        keys, tones, color_to_key, key_color, key_txt_color = results
        key_sounds = get_or_create_key_sounds(
           wav_path, framerate_hz, channels, tones, clear_cache, keys
        )
        clear_cache = True

        midi_in.set_callback(handle_midi_input, data=sounds)

        going = True
        GPIO.add_event_detect(channel, GPIO.FALLING)
        ready()
        while going:
            # When the record button is pressed leave playing loop
            if GPIO.event_detected(channel):
                GPIO.remove_event_detect(channel)
                record_sound()
                going=False


def handle_midi_input(event, sounds=None):
    """
    Handle MIDI input: play or stop sound based on MIDI message
    Parameters:
        event -- MIDI input event
        sounds -- List of sounds that can be played
    """
    message, deltatime = event
    print("len " + str(len(sounds)))
    print(f'message: {message}, deltatime: {deltatime}')
    index = message[1] - 69 + 23
    print("i " + str(index))
    if message[0] == 144 and message[2] > 2:
        pygame.mixer.Sound.play(sounds[index])
    elif message[0] == 128:
        pygame.mixer.Sound.stop(sounds[index])


def print_device_info():
    """Helper function to print info on connected midi devices"""
    pygame.midi.init()
    _print_device_info()
    pygame.midi.quit()


def _print_device_info():
    """Helper function to print info on connected midi devices"""
    for i in range(pygame.midi.get_count()):
        r = pygame.midi.get_device_info(i)
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


if __name__ == "__main__":
    play_pianoputer()
