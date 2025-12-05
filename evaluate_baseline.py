import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import math
import glob
import os
import mir_eval
import argparse
import librosa

# --- Copied functions from Auto_Correlation_Assignment_Ryan_Baker_and_Jiayi_Wang.py ---

def block_audio(audio_input, sr=None, frame_size=2048, hop_ratio=0.5, pad=True):
    # Handle input type
    if isinstance(audio_input, str):
        sr, audio_input = wavfile.read(audio_input)
    elif sr is None:
        raise ValueError("Must provide sampling rate with numpy array or file path as a string")

    # Convert to float in range [-1, 1) if needed (assuming int input from wavfile)
    if audio_input.dtype != np.float32 and audio_input.dtype != np.float64:
        if audio_input.dtype == np.uint8:
            nbits = 8
        elif audio_input.dtype == np.int16:
            nbits = 16
        elif audio_input.dtype == np.int32:
            nbits = 32
        else:
            # Fallback for other types
            nbits = 16 
        audio_input = audio_input / float(2**(nbits - 1))

    # Convert to mono if stereo
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1)

    # Calculate hop size as integer
    hop_size = int(hop_ratio * frame_size)

    # Calculate number of frames
    if pad:
        num_blocks = math.ceil((len(audio_input) - frame_size) / hop_size) + 1
        num_blocks = max(1, num_blocks)
    else:
        num_blocks = max(0, (len(audio_input) - frame_size) // hop_size + 1)

    # Initialize output arrays
    audio_blocks = np.zeros([num_blocks, frame_size])
    times = (np.arange(0, num_blocks) * hop_size) / sr

    # Extract frames
    for n in range(num_blocks):
        i_start = n * hop_size
        i_stop = i_start + frame_size
        if i_stop <= len(audio_input):
            audio_blocks[n] = audio_input[i_start:i_stop]
        else:
            remaining_samples = len(audio_input) - i_start
            if remaining_samples > 0:
                audio_blocks[n, :remaining_samples] = audio_input[i_start:]

    return audio_blocks, times

def estimate_f0(audio_frame, sr, minfreq=20, maxfreq=None, threshold=0.25):
    if maxfreq is None: maxfreq = sr / 8
    if maxfreq == 0: raise ValueError('Max Freq cannot be 0')
    if minfreq == 0: raise ValueError('Min Freq cannot be 0')
    
    f0 = np.nan
    
    # Avoid division by zero if frame is silent
    if np.max(np.abs(audio_frame)) == 0:
        return f0
        
    audio_frame = audio_frame / max(np.abs(audio_frame)) # added abs for safety
    audio_frame = audio_frame * sig.windows.blackmanharris(len(audio_frame), sym=False)
    sig.detrend(audio_frame, type='constant', overwrite_data=True)

    Tmax = 1/minfreq
    Tmin = 1/maxfreq
    Nmax = int(np.ceil(Tmax * sr))
    Nmin = int(np.floor(Tmin * sr))
    
    # Safety check for bounds
    if Nmax > len(audio_frame):
        Nmax = len(audio_frame) - 1

    corr = np.correlate(audio_frame, audio_frame, mode='full')
    corr = corr[len(audio_frame)-1:]
    
    if corr[0] == 0:
        return f0
        
    corr = corr / corr[0]

    # Check bounds for slicing
    if Nmin >= len(corr) or Nmin > Nmax:
        return f0
        
    slice_corr = corr[Nmin:Nmax+1]
    
    peak_indeces, props = sig.find_peaks(slice_corr, height=threshold)
    
    if len(peak_indeces) != 0:
        # peak_indeces are relative to slice start (Nmin)
        strongest_peak_idx = np.argmax(props["peak_heights"])
        strongest_peak = peak_indeces[strongest_peak_idx]
        
        k = Nmin + strongest_peak
        f0 = sr / k
        
    return f0

# --- Evaluation Loop ---

def load_f0_file(path):
    if not os.path.exists(path):
        return None, None
    data = np.loadtxt(path)
    times = data[:, 0]
    f0_hz = data[:, 2]
    return times, f0_hz

def evaluate_baseline(args):
    # Find paired files
    wav_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    paired_files = []
    
    for wav_path in wav_files:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        # Adjust extension if needed based on data folder structure
        f0_path = os.path.join(args.data_dir, basename + ".f0.Corrected.txt")
        if os.path.exists(f0_path):
            paired_files.append((wav_path, f0_path))
            
    # Use deterministic split to match 'val' set if desired
    import random
    random.seed(42)
    random.shuffle(paired_files)
    
    split_idx = int(len(paired_files) * 0.8)
    if args.split == 'val':
        eval_files = paired_files[split_idx:]
    elif args.split == 'train':
        eval_files = paired_files[:split_idx]
    else:
        eval_files = paired_files
        
    print(f"Evaluating Baseline on {len(eval_files)} files ({args.split})...")
    
    all_metrics = []
    
    for wav_path, ref_path in eval_files:
        # Load audio using librosa to force consistency if desired, or use wavfile as in baseline
        # Baseline uses wavfile.read inside block_audio.
        # But block_audio can take numpy array. 
        # Let's use wavfile.read via the function to be faithful to baseline logic.
        
        try:
            # Baseline parameters
            # block_audio defaults: frame_size=2048, hop_ratio=0.5
            # estimate_f0 defaults: threshold=0.25
            
            # Note: We need the sr from the file to handle time conversion correctly.
            sr_native, _ = wavfile.read(wav_path)
            
            audio_blocks, est_times = block_audio(wav_path, frame_size=2048, hop_ratio=0.5)
            
            # Predict
            f0s = []
            for frame in audio_blocks:
                val = estimate_f0(frame, sr_native, minfreq=32.7, maxfreq=2000, threshold=0.25)
                f0s.append(val)
            
            est_freqs = np.array(f0s)
            
            # Replace NaNs with 0 (unvoiced)
            est_freqs = np.nan_to_num(est_freqs, nan=0.0)
            
            # Load Reference
            ref_times, ref_freqs = load_f0_file(ref_path)
            
            # Resample Estimate to Reference times using interpolation 
            # (zero-order hold or linear? Linear is fine for continuous pitch)
            est_freqs_resampled = np.interp(ref_times, est_times, est_freqs, left=0.0, right=0.0)
            
            # Compute Metrics
            metrics = mir_eval.melody.evaluate(ref_times, ref_freqs, ref_times, est_freqs_resampled)
            all_metrics.append(metrics)
            
            print(f"{os.path.basename(wav_path)}: RPA {metrics['Raw Pitch Accuracy']:.3f}, VR {metrics['Voicing Recall']:.3f}")
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            
    if all_metrics:
        avg_metrics = {}
        for k in all_metrics[0].keys():
            avg_metrics[k] = np.mean([m[k] for m in all_metrics])
            
        print("\n=== Baseline Overall Results ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="trainData")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    
    evaluate_baseline(args)
