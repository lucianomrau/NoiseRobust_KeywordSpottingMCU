from torch.utils.data import Dataset,ConcatDataset
import torchaudio, torch
import os, glob, re, tempfile, random, shutil

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



class AudioDataset(Dataset):
    def __init__(self, audio_path, audio_list,sample_rate,duration_seconds,keywords,device):
        self.all_keywords = sorted(keywords)
        self._device = device
        self.keywords = sorted([word for word in keywords if word not in ['silence', 'unknown']])
        self.__audio_list = audio_list
        self._audio_path = [f"{str(audio_path)}{audiofile}" for audiofile in audio_list]
        self._num_samples = int(duration_seconds*sample_rate)
        self._sample_rate = sample_rate

        

    #len(audio_wav_file)
    def __len__(self):
        return len(self.__audio_list)
    
    #a_list[1] -> a_list.__getitem__(1)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label = self._encode_label(label)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = signal.to(self._device)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal, label
    
    def _resample_if_necessary(self, signal, sr):
        if sr!= self._sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self._sample_rate)
            signal = resampler(signal)
        return signal
    

    def _encode_label(self,label):
        return torch.tensor(self.all_keywords.index(label))


    def _get_audio_sample_path(self,index):
        path = self._audio_path[index]
        return path

    

    def _get_audio_sample_label(self,index):
        path = self._audio_path[index]
        parts = path.split(os.path.sep)
        label = parts[-2]

        if label == '_silence_' or label == 'silence':
            return 'silence'
        elif label == '_unkown_':
            return 'unknown'
        elif label in self.keywords:
            return label
        else:
            return 'unknown'
        
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self._num_samples:
            signal = signal[:, :self._num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self._num_samples:
            num_missing_samples = self._num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal



class SpeechCommandDataset():
    def __init__(self, audio_dataset_path, 
                 audio_test_path,
                 sample_rate,
                 duration_seconds,
                 keywords,
                 device):
        self._dataset_path = audio_dataset_path
        self._testset_path = audio_test_path
        self._device = device
        self._all_keywords = keywords
        self.keywords = [word for word in keywords if word not in ['silence', 'unknown']]
        self._train_list, self._validation_list, self.__test_list = self.__load_audio_sets()

        self.__silence_folder = self.__create_silence_samples() # for train and validation 
        # load silence files
        silence_files=os.listdir(self.__silence_folder)
        self._silence_train_list = silence_files[:self.num_silence_samples_train]
        self._silence_validation_list = silence_files[self.num_silence_samples_train+1::]
        self._sample_rate = sample_rate
        self._duration_seconds = duration_seconds



    def get_train_list(self):
        silence_files = [f"silence/{filename}" for filename in self._silence_train_list]
        return self._train_list + silence_files
    
    def get_validation_list(self):
        silence_files = [f"silence/{filename}" for filename in self._silence_validation_list]
        return self._validation_list + silence_files
    
    def get_test_list(self):
        return self.__test_list
    
    def get_silence_temp_folder(self):
        return self.__silence_folder

    def __create_silence_samples(self):
        temp_dir = tempfile.mkdtemp()

        background_noise_folder = os.path.join(self._dataset_path, '_background_noise_')

        wav_files = [os.path.join(background_noise_folder, file) for file in os.listdir(background_noise_folder) if file.endswith('.wav')]

        for idx in range(self.num_silence_samples_train+self.num_silence_samples_validation):
        # for idx in range(int((self.num_silence_samples_train+self.num_silence_samples_validation)/10)):
            random_wav_file = random.choice(wav_files)
            waveform, sample_rate = torchaudio.load(random_wav_file)

            audio_length = waveform.size(dim=1)
            start_point = random.randint(0, audio_length - sample_rate-1)

            one_second_clip = waveform[:, start_point:start_point + sample_rate]

            path = f"{temp_dir}/one_second_clip_{idx}.wav"
            torchaudio.save(path, one_second_clip, sample_rate)

    # move all the silence files into the silence directory
        destination_dir = os.path.join(temp_dir, 'silence')

        # Check if the destination directory exists
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Move all files from the source directory to the new subfolder
        for filename in os.listdir(temp_dir):
            # Create full file paths
            source_file = os.path.join(temp_dir, filename)
            destination_file = os.path.join(destination_dir, filename)

            # Check if it's a file (not a directory) before moving
            if os.path.isfile(source_file):
                shutil.move(source_file, destination_file)

        return destination_dir+"/"

    # generate the list of for each set
    def __load_audio_sets(self):
        TEST_LIST = "testing_list.txt"
        VALIDATION_LIST = "validation_list.txt"

        all_files = glob.glob(self._dataset_path + '*/*.wav')
        all_files = [os.path.relpath(f, self._dataset_path) for f in all_files]

        # Step 1: Read the paths from the text files into lists
        with open(self._dataset_path+(VALIDATION_LIST), 'r') as f:
            validation_audio_list = f.read().splitlines()

        with open(self._dataset_path+TEST_LIST, 'r') as f:
            testing_audio_list_full = f.read().splitlines()

        with open(self._testset_path+TEST_LIST, 'r') as f:
            testing_audio_list = f.read().splitlines()

        # Step 2: Generate the train list
        train_audio_list = list(set(all_files)-set(validation_audio_list)-set(testing_audio_list_full))
        train_audio_list = [item for item in train_audio_list if not re.search(r"_background_noise_", item)]

        # Step 3: filtered out the 'unknown' samples for training and validation
        train_audio_list , self.num_silence_samples_train = self.__select_unknown_samples(train_audio_list)
        validation_audio_list , self.num_silence_samples_validation = self.__select_unknown_samples(validation_audio_list)


        return train_audio_list, validation_audio_list, testing_audio_list

    
    # select randomly 'unkwown' samples. It tooks the average number of samples from the 10 keywords
    def __select_unknown_samples(self,audio_list):
        # Count occurrences of each keyword
        keyword_counts = {}
        for item in audio_list:
            parts = item.split(os.path.sep)
            label = parts[-2]
            if label in self.keywords:
                keyword_counts[label] = keyword_counts.get(label, 0) + 1
        
        # Calculate average number of samples per keyword
        avg_samples = round(sum(keyword_counts.values()) / len(keyword_counts)) if keyword_counts else 0
        
        # Identify samples not in self.keywords
        non_keyword_samples = [item for item in audio_list 
                                if item.split(os.path.sep)[-2] not in self.keywords]
        
        # Randomly select samples from non-keyword samples
        random_samples = random.sample(non_keyword_samples, 
                                    min(int(avg_samples), len(non_keyword_samples)))
        
        # Create final filtered list
        final_list = [item for item in audio_list 
                    if (item.split(os.path.sep)[-2] in self.keywords or 
                        item in random_samples)]
        
        return final_list,avg_samples

    def __create_dataset(self, file_list, path):
        return AudioDataset(path , file_list,self._sample_rate,self._duration_seconds,self._all_keywords,self._device)
    
    def get_trainset(self):
        train_files = self._train_list
        path = self._dataset_path
        keywordsDataset = self.__create_dataset(train_files , path)
        silenceDataset = self.__create_dataset(self._silence_train_list , self.get_silence_temp_folder())
        return ConcatDataset([keywordsDataset, silenceDataset])

    def get_validationset(self):
        val_files = self._validation_list
        path = self._dataset_path
        keywordsDataset = self.__create_dataset(val_files , path)
        silenceDataset = self.__create_dataset(self._silence_validation_list , self.get_silence_temp_folder())
        return ConcatDataset([keywordsDataset, silenceDataset])

    def get_testset(self):
        test_files = self.__test_list
        path = self._testset_path
        return self.__create_dataset(test_files, path)