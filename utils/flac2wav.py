import librosa as rosa
from scipy.io.wavfile import write as swrite
from omegaconf import OmegaConf as OC
import os
from glob import glob
from tqdm import tqdm
import multiprocessing as mp


def flac2wav(wav):
    y,_ = rosa.load(wav, sr = hparams.audio.sampling_rate, mono = True)
    file_id = os.path.split(wav)[-1].split('_mic')[0]
    if file_id in timestamps:
        start, end = timestamps[file_id]
        start = start - min(start, int(0.1 * hparams.audio.sampling_rate))
        end = end + min(len(y) - end, int(0.1 * hparams.audio.sampling_rate))
        y = y[start:end]

    os.makedirs(os.path.join(hparams.data.dir, wav.split(os.sep)[-2]), exist_ok=True)

    wav_path = os.path.splitext(os.path.join(hparams.data.dir, wav.split(os.sep)[-2], os.path.split(wav)[-1]))[0]+'.wav'

    swrite(wav_path, hparams.audio.sampling_rate, y)
    del y
    return

if __name__=='__main__':
    hparams = OC.load('hparameter.yaml')
    base_dir = hparams.data.base_dir
    os.makedirs(hparams.data.dir, exist_ok=True)

    wavs = glob(os.path.join(base_dir, '*/*.flac'))

    timestamps = {}
    path_timestamps = hparams.data.timestamp_path
    with open(path_timestamps, 'r') as f:
        timestamps_list = f.readlines()
    for line in timestamps_list:
        timestamp_data = line.strip().split(' ')
        if len(timestamp_data) == 3:
            file_id, t_start, t_end = timestamp_data
            t_start = int(float(t_start) * hparams.audio.sampling_rate)
            t_end = int(float(t_end) * hparams.audio.sampling_rate)
            timestamps[file_id] = (t_start, t_end)

    pool = mp.Pool(processes = hparams.train.num_workers)
    with tqdm(total = len(wavs)) as pbar:
        for _ in tqdm(pool.imap_unordered(flac2wav, wavs)):
            pbar.update()
