import os
import numpy as np
import torch
import torch.utils.data

from rvc.v2_core.vendor.mel_processing import spectrogram_torch
from rvc.v2_core.vendor.utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    Dataset that loads text and audio pairs.

    Args:
        hparams: Hyperparameters.
    """

    def __init__(self, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self._filter()

    def _filter(self):
        """
        Filters audio paths and text pairs based on text length.
        """
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        """
        Converts speaker ID to a LongTensor.

        Args:
            sid (str): Speaker ID.
        """
        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError as error:
            print(f"Error converting speaker ID '{sid}' to integer. Exception: {error}")
            sid = torch.LongTensor([0])
        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        """
        Loads and processes audio and text data for a single pair.

        Args:
            audiopath_and_text (list): List containing audio path, text, pitch, pitchf, and speaker ID.
        """
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        pitch = audiopath_and_text[2]
        pitchf = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        len_phone = phone.size()[0]
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]
            phone = phone[:len_min]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        phone = np.load(phone)
        pitch = np.load(pitch)
        pitchf = np.load(pitchf)
        assert phone.shape[-1] == pitch.shape[-1] == pitchf.shape[-1]

        return (
            torch.LongTensor(phone),  # int64
            torch.LongTensor(pitch),  # int64
            torch.FloatTensor(pitchf),  # float32
        )

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        assert sampling_rate == self.sample_rate

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return (spec, audio_norm)

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """
    Collate function for creating batches of data.
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, y, _, _, _, _ = list(zip(*batch))
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in y]), dim=0, descending=True
        )
        max_y_len = max([x.size(1) for x in y])

        def reprocess(batch, idxs):
            spec, y, phone, pitch, pitchf, dv = list(zip(*[batch[idx] for idx in idxs]))
            spec = torch.stack(spec)
            phone = torch.stack(phone)
            pitch = torch.stack(pitch)
            pitchf = torch.stack(pitchf)
            dv = torch.stack(dv)

            y = [y[i] for i in range(len(y))]
            y_mel = torch.zeros(len(y), 1, max_y_len)
            for i in range(len(y)):
                y_mel[i, :, : y[i].shape[1]] = y[i]

            len_phone = torch.tensor([phone[i].size(0) for i in range(len(phone))])
            len_spec = torch.tensor([spec[i].size(1) for i in range(len(spec))])
            len_y = torch.tensor([y[i].shape[1] for i in range(len(y))])

            return (
                spec,
                y_mel,
                phone,
                pitch,
                pitchf,
                dv,
                len_phone,
                len_spec,
                len_y,
            )

        out = reprocess(batch, ids_sorted_decreasing)
        if self.return_ids:
            return out + (ids_sorted_decreasing,)
        return out


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=False,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.boundaries = boundaries
        self.batch_size = batch_size
        self.buckets, self.num_samples_per_bucket = self._create_buckets(dataset)
        self.batch_count = [int(i) for i in self.num_samples_per_bucket]
        self.nonused_indices = []
        self.reset_flag = True

    def _create_buckets(self, dataset):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(dataset)):
            length = dataset.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket == -1:
                continue
            buckets[idx_bucket].append(i)
        max_bucket_len = max([len(bucket) for bucket in buckets])
        for i in range(len(buckets)):
            if len(buckets[i]) == 0:
                buckets[i] = [0]
                max_bucket_len = max(max_bucket_len, 1)
        total_batch = 0
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            rem = (self.batch_size - (len_bucket % self.batch_size)) % self.batch_size
            buckets[i].extend(buckets[i][:rem])
            num_batches = (len_bucket + rem) // self.batch_size
            total_batch += num_batches
            num_samples_per_bucket.append(num_batches)
        for i in range(len(buckets)):
            buckets[i] = torch.LongTensor(buckets[i])
        return buckets, num_samples_per_bucket

    def __iter__(self):
        if self.reset_flag:
            self.epoch = 0
            self.reset_flag = False
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            for i in range(len(self.buckets)):
                self.buckets[i] = self.buckets[i][torch.randperm(len(self.buckets[i]), generator=g)]
        self.batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            self.batch_count[i] = 0
            for j in range(len(bucket) // self.batch_size):
                self.batches.append(bucket[j * self.batch_size : (j + 1) * self.batch_size])
                self.batch_count[i] += 1
        if self.shuffle:
            self.batches = [self.batches[i] for i in torch.randperm(len(self.batches), generator=g)]
        else:
            self.batches = [self.batches[i] for i in range(len(self.batches))]
        self.nonused_indices = []
        for i in self.batches:
            indices = i.tolist()
            for j in indices:
                self.nonused_indices.append(j)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.reset_flag = True

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

