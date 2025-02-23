import torch
import torchaudio
import numpy as np
from speech_command_dataset import ProcessAudio
import scipy.signal as signal

SEED = 42
g_cpu = torch.Generator()
g_cuda = torch.Generator(device='cuda')
g_cpu.manual_seed(SEED)
torch.manual_seed(SEED)
g_cuda.manual_seed(SEED)

class AddNoise(ProcessAudio):
    """
    This class mix signal and noise at a specific SNR value.
    The mixing is additive.
    TODO: Add other mix scenarios: convolutive, reverberance, etc.
    """
    def __init__(self,noise_type,snr_db,transformation,device,sample_rate,duration_seconds,dataset):
        num_samples = int(duration_seconds*sample_rate)
        
        super().__init__(sample_rate,num_samples)
        self.noise_type = noise_type
        self.snr_db = snr_db
        self._device = device
        # self._sample_rate = sample_rate
        
        self.mel_spectrogram = transformation.to(device)
        # white, pink, babble and classes from the UrbanSoundsDataset
        self.noise_type_allowed = ['white',
                                   'pink',
                                   'babble',
                                   'air_conditioner_background',
                                   'car_horn_background',
                                   'children_playing_background',
                                   'dog_bark_background',
                                    'drilling_background',
                                    'engine_idling_background',
                                    'gun_shot_background',
                                    'jackhammer_background',
                                    'siren_background',
                                    'street_music_background',
                                    'air_conditioner_foreground',
                                   'car_horn_foreground',
                                   'children_playing_foreground',
                                   'dog_bark_foreground',
                                    'drilling_foreground',
                                    'engine_idling_foreground',
                                    'gun_shot_foreground',
                                    'jackhammer_foreground',
                                    'siren_foreground',
                                    'street_music_foreground']
        assert self.noise_type in self.noise_type_allowed, "Noise type not allowed"  # TODO: implement more noise types
        self.noise_signal = self._load_noise_signal()
        self.dataset = dataset
    
    def _load_noise_signal(self):

        if self.noise_type == 'white':
            noise_signal = self.__white_noise()
        elif self.noise_type == 'pink':
            noise_signal = self.__pink_noise()
        elif self.noise_type == 'babble':
            noise_signal = self.__babble_noise()
        else:
            noise_signal = self.__urban_sounds_noise()
        return noise_signal
    
    def __white_noise(self):
        noise = torch.randn(self._num_samples,device=self._device,generator=g_cuda)
        return noise
    
    def __pink_noise(self):
        # Generate white noise
        white_noise = torch.randn(self._num_samples,generator=g_cpu)

        # Apply a filter to shape the white noise into pink noise
        b, a = signal.butter(4, 0.05, 'highpass')  # High-pass filter to remove DC component
        filtered_noise = signal.lfilter(b, a, white_noise)
        
        fft = torch.fft.rfft(torch.tensor(filtered_noise))
        frequencies = torch.fft.rfftfreq(self._num_samples, d=1/self._sample_rate)
        pink_filter = 1 / torch.sqrt(np.abs(frequencies + 1e-10))  # Avoid division by zero
        pink_filter[0] = 0  # Remove DC component
        pink_fft = fft * pink_filter

        # Inverse FFT to get the time-domain signal
        pink_noise = torch.fft.irfft(pink_fft)
        pink_noise = pink_noise.to(torch.float32)

        return pink_noise.to(self._device)
    
    def __babble_noise(self):
        noise,sr = torchaudio.load("./noise/noisex-92/babble.wav")
        
        noise = self._resample_if_necessary(noise, sr)
        noise = noise.to(self._device)
        noise = self._cut_if_necessary(noise)
        noise = self._right_pad_if_necessary(noise)
        return noise

    def __urban_sounds_noise(self): #TODO: read the wav file name from the csv file
        wav_dict = {
                    'air_conditioner_background' : "177621-0-0-0.wav",
                    'car_horn_background' : "132073-1-0-0.wav",
                    'children_playing_background' : "135776-2-0-32.wav",
                    'dog_bark_background' : "102106-3-0-0.wav",
                    'drilling_background' : "17913-4-1-0.wav",
                    'engine_idling_background' : "46918-5-0-0.wav",
                    'gun_shot_background' : "135527-6-0-0.wav",
                    'jackhammer_background' : "180937-7-3-0.wav",
                    'siren_background' : "106905-8-0-0.wav",
                    'street_music_background' : "132016-9-0-0.wav",
                    'air_conditioner_foreground' : "127873-0-0-0.wav",
                    'car_horn_foreground' : "145577-1-0-0.wav",
                    'children_playing_foreground' : "105415-2-0-1.wav",
                    'dog_bark_foreground' : "101415-3-0-2.wav",
                    'drilling_foreground' : "103199-4-0-0.wav",
                    'engine_idling_foreground' : "103258-5-0-0.wav",
                    'gun_shot_foreground' : "102305-6-0-0.wav",
                    'jackhammer_foreground' : "103074-7-0-0.wav",
                    'siren_foreground' : "157867-8-0-0.wav",
                    'street_music_foreground' : "108041-9-0-11.wav"
                    }
        
        noise,sr = torchaudio.load("./noise/UrbanSound8K/" + wav_dict[self.noise_type])
        
        noise = self._resample_if_necessary(noise, sr)
        noise = noise.to(self._device)
        noise = self._cut_if_necessary(noise)
        noise = self._right_pad_if_necessary(noise)
        noise = self._mix_down_if_necessary(noise)
        return noise



    def aditive_noise(self,signal):
        # Add noise to the signal
        signal_with_noise = torchaudio.functional.add_noise(waveform=torch.squeeze(signal),
                                                            noise=torch.squeeze(self.noise_signal),
                                                            snr=torch.tensor(self.snr_db),
                                                            lengths=None)
        signal_with_noise = torch.unsqueeze(signal_with_noise, 0)
        mel_spectro = self.mel_spectrogram(signal_with_noise)
        # Convert to decibels
        mel_spectro = torchaudio.transforms.AmplitudeToDB()(mel_spectro)
        return mel_spectro

    def __getitem__(self, idx):
        waveform, label = self.dataset[idx]
        mel_spectrogram = self.aditive_noise(waveform)
        return mel_spectrogram, label
        