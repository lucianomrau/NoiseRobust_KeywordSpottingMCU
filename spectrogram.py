import torchaudio
import os
import subprocess
from torch.utils.data import ConcatDataset

initial_dir = os.getcwd()
PATH_SPEECH_COMMAND_DATASET = os.path.join(initial_dir , 'speech_command_dataset.py' )

if not(os.path.exists(PATH_SPEECH_COMMAND_DATASET)):
    subprocess.run(["wget", "https://raw.githubusercontent.com/lucianomrau/Use_SpeechCommandDatasetV2_properly/refs/heads/master/speech_command_dataset.py"])
    
from speech_command_dataset import SpeechCommandDataset, AudioDataset


class SpectrogramAudioDataset(AudioDataset):
    def __init__(self, audio_path, audio_list, 
                 n_mels, n_fft,hop_length,
                 win_length,f_min,f_max,
                 power,sample_rate,duration_seconds,
                 keywords, device):
        super().__init__(audio_path, audio_list,sample_rate,duration_seconds,keywords,device)
        transformation = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power = power,
            sample_rate=sample_rate,
        )
        self.mel_spectrogram = transformation.to(self._device)


    def __getitem__(self, index):
        signal, label = super().__getitem__(index)
        mel_spectro = self.mel_spectrogram(signal)
        # Convert to decibels
        mel_spectro = torchaudio.transforms.AmplitudeToDB()(mel_spectro)
        return mel_spectro, label

class SpectrogramSpeechCommandDataset(SpeechCommandDataset):
    def __init__(self, audio_dataset_path, audio_test_path, n_mels, n_fft, 
                 hop_length,win_length,f_min,f_max,power,sample_rate,duration_seconds,keywords,device):
        
        super().__init__(audio_dataset_path, audio_test_path,sample_rate,duration_seconds,keywords,device)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power

    def get_spectrum_trainset(self):
        train_files = self._train_list
        path = self._dataset_path
        keywordsDataset = self.__create_spectrogram_dataset(train_files, path)
        silenceDataset = self.__create_spectrogram_dataset(
            self._silence_train_list,
            self.get_silence_temp_folder()
        )
        return ConcatDataset([keywordsDataset, silenceDataset])

    def get_spectrum_validationset(self):
        val_files = self._validation_list
        path = self._dataset_path
        keywordsDataset = self.__create_spectrogram_dataset(val_files, path)
        # print("silence")
        # print(self._silence_validation_list)
        silenceDataset = self.__create_spectrogram_dataset(
            self._silence_validation_list,
            self.get_silence_temp_folder()
        )
        # print(silenceDataset[0])
        return ConcatDataset([keywordsDataset, silenceDataset])

    def get_spectrum_testset(self):
        test_files = self.get_test_list()
        path = self._testset_path
        return self.__create_spectrogram_dataset(test_files, path)

    def __create_spectrogram_dataset(self, file_list, path):
        return SpectrogramAudioDataset(
            path, 
            file_list, 
            self.n_mels, 
            self.n_fft, 
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            self.power,
            self._sample_rate,
            self._duration_seconds,
            self._all_keywords,
            self._device
        )