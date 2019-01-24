"""
Base code: https://github.com/RayanWang/Speech_emotion_recognition_BLSTM
It must be modified to make it work correctly.

There is no universal solution for all audios.
You have to play with the following variables:

-chunk_duration_ms  (en la llamada desde '__main__')
-self._vad = webrtcvad.Vad(mode=3)
-num_voiced > 0.8
-start_point = index - self._chunk_size * 17
-num_unvoiced > 0.70
-end_point = index - self._nb_window_chunks_end * 240
"""

from optparse import OptionParser

from array import array
from struct import pack

from utility import audiosegment
import os
import glob
import webrtcvad
import sys
import wave


def record_to_file(path, data, sample_width, sr=16000):
    """
    Records from the wav utility and outputs the resulting data to 'path'
    """
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(sr)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    """
    Average the volume out
    """
    max_value = max(abs(i) for i in snd_data)
    if max_value == 0:
        return snd_data
    maximum = 32767
    times = float(maximum) / max_value
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


class AudioPreprocessing(object):

    def __init__(self, sr=16000, chunk_duration_ms=30, wav_path='', out_path=''):
        self._sr = sr
        self._chunk_duration_ms = chunk_duration_ms
        self._chunk_size = int(sr * chunk_duration_ms / 1000)  # chunk to read in samples
        self._nb_window_chunks = int(400 / chunk_duration_ms)  # 400ms / 30ms frame
        self._nb_window_chunks_end = self._nb_window_chunks * 2
        self._vad = webrtcvad.Vad(mode=3)

        self.wav_path = wav_path
        self._out_path = out_path

    def sentence_slicing(self, filename):
        print("Re-sampling...          ->   " + filename)
        seg = audiosegment.from_file(filename).resample(sample_rate_Hz=self._sr, sample_width=2, channels=1)

        print("\tDetecting voice...")

        got_sentence = False
        ended = False
        offset = self._chunk_duration_ms

        end_point = 0

        path = filename.split('/')[-1]
        path = self._out_path + '/' + path

        i = 1
        while not ended:
            triggered = False
            buffer_flags = [0] * self._nb_window_chunks
            buffer_index = 0

            buffer_flags_end = [0] * self._nb_window_chunks_end
            buffer_index_end = 0

            raw_data = array('h')
            index = 0
            start_point = 0

            while not got_sentence and not ended:
                chunk = seg[(offset - self._chunk_duration_ms):offset].raw_data

                raw_data.extend(array('h', chunk))
                offset += self._chunk_duration_ms
                index += self._chunk_size

                active = self._vad.is_speech(chunk, self._sr)

                buffer_flags[buffer_index] = 1 if active else 0
                buffer_index += 1
                buffer_index %= self._nb_window_chunks

                buffer_flags_end[buffer_index_end] = 1 if active else 0
                buffer_index_end += 1
                buffer_index_end %= self._nb_window_chunks_end

                # start point detection
                if not triggered:
                    num_voiced = sum(buffer_flags)
                    if num_voiced > 0.8 * self._nb_window_chunks:
                        # sys.stdout.write(' Start sentence ')
                        triggered = True
                        start_point = index - self._chunk_size * 17  # start point
                # end point detection
                else:
                    num_unvoiced = self._nb_window_chunks_end - sum(buffer_flags_end)
                    if num_unvoiced > 0.70 * self._nb_window_chunks_end:
                        # sys.stdout.write(' End sentence ')
                        triggered = False
                        got_sentence = True
                        end_point = index - self._nb_window_chunks_end * 240  # end point

                if offset >= len(seg):
                    sys.stdout.write(' File end ')
                    ended = True

                sys.stdout.flush()

            sys.stdout.write('\n')

            got_sentence = False

            print('\t\tStart point: %d' % start_point)

            if end_point != 0:
                raw_data = raw_data[:end_point]
                print('\t\tEnd point: %d' % end_point)
            else:
                print("\t\tEnd point: NO")

            # write to file
            raw_data.reverse()
            for _ in range(start_point):
                raw_data.pop()

            raw_data.reverse()

            raw_data = normalize(raw_data)

            print('\tSentence length: %d bytes' % (len(raw_data) * 2))

            f, ext = os.path.splitext(path)

            record = 0
            if i != 1:
                print("merda=" + str(i))
                f = f + '_' + str(i) + ext
            else:
                f = f + ext
                record = 1
            i += 1

            if record == 1:
                record_to_file(f, raw_data, 2)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--wav_path', dest='path', default='')
    parser.add_option('-d', '--output_dir', dest='dir', default='')

    (options, args) = parser.parse_args(sys.argv)

    path = options.path
    out_dir = options.dir

    audioprocessing = AudioPreprocessing(wav_path=path, out_path=out_dir, chunk_duration_ms=20)
    for wav in glob.glob(path + '/*.wav'):
        audioprocessing.sentence_slicing(wav)
