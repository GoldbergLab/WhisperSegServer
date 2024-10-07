import glob
import os
import librosa
fd_wseg = '/home/glab/Code/WhisperSeg/'
os.chdir(fd_wseg)
from model import WhisperSegmenterFast, WhisperSegmenter

def ZZ_getSegFolder_v1(fd_wav, fn_ckpt, param):
    # apply trained model to a folder of wav files to get segmentation
    segmenter = WhisperSegmenterFast(fn_ckpt, device="cuda" )
    print(f'Using model in: {fn_ckpt}')
    fns_wav = sorted(glob.glob(os.path.join(fd_wav, '*.wav')))
    pred_all = []
    for audio_file in fns_wav:
        audio, _ = librosa.load( audio_file, sr = param['sr'] )
        prediction = segmenter.segment( audio, sr = param['sr'], min_frequency = param['min_frequency'], spec_time_step = param['spec_time_step'],
                               min_segment_length = param['min_segment_length'], eps = param['eps'], num_trials = param['num_trials'])
        pred_all.append(prediction)
    return fns_wav, pred_all

# parameter for WhisperSeg
param = {'sr': 32000, 'min_frequency': 0, 'spec_time_step': 0.0025, 'min_segment_length': 0.01, 'tolerance': 0.01,
         'time_per_frame_for_scoring': 0.001, 'eps': 0.02, 'num_trials': 3}

# get the segmentation of a folder of wav files
fd_test = '/home/glab/Downloads/wseg_test'
fn_ckpt = '/home/glab/Downloads/final_checkpoint_ct2'
fns_wav, pred_all = ZZ_getSegFolder_v1(fd_test, fn_ckpt, param)
print(pred_all)
