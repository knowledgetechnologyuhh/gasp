# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Fares Abawi (fares.abawi@uni-hamburg.de)
# ==============================================================================


import resampy
import numpy as np

from gazenet.utils.registrar import *


@AudioFeatureRegistrar.register
class WindowedAudioFeatures(object):
    """Split the raw signal into windows and applies a hanning window without any spectral transformation"""
    def __init__(self, rate=48000,
                 win_len=64,
                 **kwargs):
        self.rate = rate
        self.win_len = win_len

    def waveform_to_feature(self, data, rate):
        # TODO (fabawi): resampling is an improvisation. Check if it actually works
        data = resampy.resample(data, rate, self.rate)
        audiowav = data * (2 ** -23)
        # TODO (fabawi): check the hanning
        audiowav = np.hanning(audiowav.shape[1]) * audiowav
        return audiowav


@AudioFeatureRegistrar.register
class MFCCAudioFeatures(object):
    """Defines routines to compute mel spectrogram features from audio waveform."""
    def __init__(self, rate=16000,
                 win_len_sec=0.64,  # Each example contains 50 10ms video_frames_list
                 hop_len_sec=0.02,  # Defined dynamically as 1/(video_fps)
                 log_offset=0.010,  # Offset used for stabilized log of input mel-spectrogram
                 stft_win_len_sec=0.025,
                 stft_hop_len_sec=0.010,
                 mel_len=64,
                 mel_min_hz=125,
                 mel_max_hz=7500,
                 mel_break_hz=700.0,
                 mel_high_q=1127.0,
                 **kwargs):
        self.rate = rate
        self.win_len_sec = win_len_sec
        self.hop_len_sec = hop_len_sec
        self.log_offset = log_offset
        self.stft_win_len_sec = stft_win_len_sec
        self.stft_hop_len_sec = stft_hop_len_sec
        self.mel_len = mel_len
        self.mel_min_hz = mel_min_hz
        self.mel_max_hz = mel_max_hz
        self.mel_break_hz = mel_break_hz
        self.mel_high_q = mel_high_q

    def frame(self, data, win_len, hop_len):
        """Convert array into a sequence of successive possibly overlapping video_frames_list.

        An n-dimensional array of shape (num_samples, ...) is converted into an
        (n+1)-D array of shape (num_frames, window_length, ...), where each frame
        starts hop_length points after the preceding one.

        This is accomplished using stride_tricks, so the original data is not
        copied.  However, there is no zero-padding, so any incomplete video_frames_list at the
        end are not included.

        Args:
          data: np.array of dimension N >= 1.
          win_len: Number of samples in each frame.
          hop_len: Advance (in samples) between each window.

        Returns:
          (N+1)-D np.array with as many rows as there are complete video_frames_list that can be
          extracted.
        """
        num_samples = data.shape[0]
        num_frames = 1 + int(np.floor((num_samples - win_len) / hop_len))
        shape = (num_frames, win_len) + data.shape[1:]
        strides = (data.strides[0] * hop_len,) + data.strides
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def periodic_hann(self, win_len):
        """Calculate a "periodic" Hann window.

        The classic Hann window is defined as a raised cosine that starts and
        ends on zero, and where every value appears twice, except the middle
        point for an odd-length window.  Matlab calls this a "symmetric" window
        and np.hanning() returns it.  However, for Fourier analysis, this
        actually represents just over one cycle of a period N-1 cosine, and
        thus is not compactly expressed on a length-N Fourier basis.  Instead,
        it's better to use a raised cosine that ends just before the final
        zero value - i.e. a complete cycle of a period-N cosine.  Matlab
        calls this a "periodic" window. This routine calculates it.

        Args:
        win_len: The number of points in the returned window.

        Returns:
        A 1D np.array containing the periodic hann window.
        """
        return 0.5 - (0.5 * np.cos(2 * np.pi / win_len * np.arange(win_len)))

    def stft_magnitude(self, signal, fft_len, hop_len, win_len):
        """Calculate the short-time Fourier transform magnitude.

        Args:
        signal: 1D np.array of the input time-domain signal
        fft_len: Size of the FFT to apply
        hop_len: Advance (in samples) between each frame passed to FFT
        win_len: Length of each block of samples to pass to FFT

        Returns:
        2D np.array where each row contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame of input samples.
        """
        frames = self.frame(signal, win_len, hop_len)
        # Apply frame window to each frame. We use a periodic Hann (cosine of period
        # window_length) instead of the symmetric Hann of np.hanning (period
        # window_length-1).
        window = self.periodic_hann(win_len)
        windowed_frames = frames * window
        return np.abs(np.fft.rfft(windowed_frames, int(fft_len)))

    def hertz_to_mel(self, freqs_hz):
        """Convert frequencies to mel scale using HTK formula.

        Args:
        freqs_hertz: Scalar or np.array of frequencies in hertz.

        Returns:
        Object of same size as frequencies_hertz containing corresponding values
        on the mel scale.
        """
        return self.mel_high_q * np.log(
            1.0 + (freqs_hz / self.mel_break_hz))

    def spectrogram_to_mel_matrix(self, spectro_len=129):

        """Return a matrix that can post-multiply spectrogram rows to make mel.

        Returns a np.array matrix A that can be used to post-multiply a matrix S of
        spectrogram values (STFT magnitudes) arranged as video_frames_list x bins to generate a
        "mel spectrogram" M of video_frames_list x mel_len.  M = S A.

        The classic HTK algorithm exploits the complementarity of adjacent mel bands
        to multiply each FFT bin by only one mel weight, then add it, with positive
        and negative signs, to the two adjacent mel bands to which that bin
        contributes.  Here, by expressing this operation as a matrix multiply, we go
        from num_fft multiplies per frame (plus around 2*num_fft adds) to around
        num_fft^2 multiplies and adds.  However, because these are all presumably
        accomplished in a single call to np.dot(), it's not clear which approach is
        faster in Python.  The matrix multiplication has the attraction of being more
        general and flexible, and much easier to read.

        Args:
        spectro_len: How many bins there are in the source spectrogram
          data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
          only contains the nonredundant FFT bins

        Returns:
        An np.array with shape (num_spectrogram_bins, mel_len).

        Raises:
        ValueError: if frequency edges are incorrectly ordered or out of range.
        """
        nyquist_hz = self.rate / 2.
        mel_len = self.mel_len
        mel_min_hz = self.mel_min_hz
        mel_max_hz = self.mel_max_hz
        if mel_min_hz < 0.0:
            raise ValueError("mel_min_hz %.1f must be >= 0" % mel_min_hz)
        if mel_min_hz >= mel_max_hz:
            raise ValueError("mel_min_hz %.1f >= mel_max_hz %.1f" %
                             (mel_min_hz, mel_max_hz))
        if mel_max_hz > nyquist_hz:
            raise ValueError("mel_max_hz %.1f is greater than Nyquist %.1f" %
                             (mel_max_hz, nyquist_hz))
        spectro_len_hz = np.linspace(0.0, nyquist_hz, spectro_len)
        spectro_len_mel = self.hertz_to_mel(spectro_len_hz)
        # The i'th mel band (starting from i=1) has center frequency
        # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
        # band_edges_mel[i+1].  Thus, we need mel_len + 2 values in
        # the band_edges_mel arrays.
        band_edges_mel = np.linspace(self.hertz_to_mel(mel_min_hz),
                                     self.hertz_to_mel(mel_max_hz), mel_len + 2)
        # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
        # of spectrogram values.
        mel_weights_matrix = np.empty((spectro_len, mel_len))
        for i in range(mel_len):
            lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
            # Calculate lower and upper slopes for every spectrogram bin.
            # Line segments are linear in the *mel* domain, not hertz.
            lower_slope = ((spectro_len_mel - lower_edge_mel) /
                           (center_mel - lower_edge_mel))
            upper_slope = ((upper_edge_mel - spectro_len_mel) /
                           (upper_edge_mel - center_mel))
            # .. then intersect them with each other and zero.
            mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                                  upper_slope))
        # HTK excludes the spectrogram DC bin; make sure it always gets a zero
        # coefficient.
        mel_weights_matrix[0, :] = 0.0
        return mel_weights_matrix

    def log_mel_spectrogram(self, data):
        """Convert waveform to a log magnitude mel-frequency spectrogram.

        Args:
        data: 1D np.array of waveform data

        Returns:
        2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
        magnitudes for successive video_frames_list.
        """
        win_len_samples = int(round(self.rate * self.stft_win_len_sec))
        hop_len_samples = int(round(self.rate * self.stft_hop_len_sec))
        fft_length = 2 ** int(np.ceil(np.log(win_len_samples) / np.log(2.0)))
        spectrogram = self.stft_magnitude(
            data,
            fft_len=fft_length,
            hop_len=hop_len_samples,
            win_len=win_len_samples)
        mel_spectrogram = np.dot(spectrogram, self.spectrogram_to_mel_matrix(
            spectro_len=spectrogram.shape[1]))
        return np.log(mel_spectrogram + self.log_offset)

    def extract_feature(self, data, rate, mono=False):
        """Converts audio waveform into an array of examples for VGGish.

        Args:
        sound: np.array of  two dimensions(multi-channel, with the
            outer dimension representing channels). Each sample is generally
            expected to lie in the range [-1.0, +1.0], although this is not required.
        sample_rate: Sample rate of data

        Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames video_frames_list of audio and num_bands mel frequency
        bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
        """
        if mono:
            data = data.flatten()[...]
        else:
            data = data.flatten()[..., np.newaxis]
        # # convert to mono.
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # resample to the rate assumed by VGGish
        if rate != self.rate:
            if rate > data.shape[0]:
                data = np.append(data, np.zeros((rate - data.shape[0],)), axis=0)
            # TODO (fabawi): set this to data.copy() which is expensive but might avoid memory pointer issues
            data = resampy.resample(data, rate, self.rate)

        # compute log mel spectrogram features
        log_mel = self.log_mel_spectrogram(data)

        # frame features into examples
        features_sample_rate = 1.0 / self.stft_hop_len_sec
        example_win_len = int(round(
            self.win_len_sec * features_sample_rate))
        example_hop_len = int(round(
            self.hop_len_sec * features_sample_rate))
        log_mel_examples = self.frame(
            log_mel,
            win_len=example_win_len,
            hop_len=example_hop_len)
        return log_mel_examples

    def waveform_to_feature(self, sound, rate):

        # TODO (fabawi): implement a better way to distinguish mono from stereo
        # if len(sound.shape) > 3:
        if len(sound.shape) > 2 and sound.shape[2] > 1:
            # if channels last, then rotate sound. This will break if no. of channels is more than the framerate but that should not happen
            if sound.shape[0] > sound.shape[2]:
                sound = np.moveaxis(sound, -1, 0)
            log_mel_examples = None
            for data in sound:
                if log_mel_examples is None:
                    log_mel_examples = self.extract_feature(data, rate)
                else:
                    log_mel_examples = np.stack((log_mel_examples, self.extract_feature(data, rate)), axis=0)
            return log_mel_examples[np.newaxis, ...]
        else:
            log_mel_examples = self.extract_feature(sound, rate, mono=True)
            return log_mel_examples

