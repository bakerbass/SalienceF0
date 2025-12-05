import numpy as np
import librosa
import math
import os
import glob
import torch
from torch.utils.data import Dataset
import random

# Global constants as per assignment/paper recommendations (and CNN_demo.ipynb)
TARGET_SR = 22050
BINS_PER_SEMITONE = 3  # High resolution
N_OCTAVES = 6
FMIN = 32.7
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
HOP_LENGTH = 512
HARMONICS = [0.5, 1, 2, 3, 4] # 5 harmonics for HCQT

def compute_hcqt(y, sr, harmonics=HARMONICS, bins_per_octave=BINS_PER_OCTAVE, n_octaves=N_OCTAVES, hop_length=HOP_LENGTH, fmin=FMIN):
    """
    Computes the Harmonic CQT (HCQT) for a given audio signal.
    Returns:
        hcqt: (H, T, F) tensor where H is number of harmonics
        freqs: Array of center frequencies
    """
    # Calculate frequencies once
    freqs = librosa.cqt_frequencies(
        n_bins=n_bins_from_octaves(n_octaves, bins_per_octave),
        fmin=fmin,
        bins_per_octave=bins_per_octave,
    )
    
    n_bins = len(freqs)
    hcqt_list = []

    for h in harmonics:
        # Pitch shift audio to align harmonics
        # Note: librosa.effects.pitch_shift is slow. 
        # Deep Salience paper actually computes CQT at different fmins.
        # However, the demo notebook used pitch_shift, so we stick to that or the fmin trick.
        # The fmin trick is much faster: compute CQT of 'y' with fmin * h
        # But we want the output bins to align.
        # If we change fmin to fmin*h, the bins correspond to frequencies fmin*h * 2^(k/bpo).
        # We want the bin 'k' to represent fundamental frequency f0.
        # So if we want to capture the h-th harmonic at bin 'k', we are looking for energy at f0 * h.
        # If we compute CQT with standard fmin, the energy at f0*h will be at a higher bin index.
        # If we strictly follow Bittner et al., they compute the CQT with *different minimum frequencies*.
        # Let's try to mimic the demo notebook's logic but maybe optimize if needed.
        # The demo notebook used pitch shifting y. That's extremely slow for training.
        # Let's use the standard "compute CQT at fmin*h" approach which is mathematically cleaner for HCQT 
        # IF we want the 0-th bin of channel h to correspond to fmin * h.
        # BUT for the stack, we usually want them aligned such that the k-th bin in ALL channels corresponds to the SAME fundamental pitch f0.
        # If channel h captures energy at h*f0, and we want it aligned with channel 1 at f0,
        # we need to shift the CQT down by log2(h).
        # OR we compute CQT with fmin = original_fmin * h.
        # Then bin 0 corresponds to freq = fmin * h. 
        # This aligns the harmonic h*f0 to the same position as f0 in the base CQT.
        
        # Implementation:
        C = librosa.cqt(
            y, sr=sr, hop_length=hop_length,
            fmin=fmin * h,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins,
        )
        Cmag = np.abs(C)
        hcqt_list.append(Cmag)

    # Stack: (H, F, T) -> Transpose to (H, T, F) if preferred, but usually (H, T, F) or (H, F, T) conventions vary.
    # The demo notebook produced (H, T, F). Librosa produces (F, T).
    # So we stack to (H, F, T) then transpose.
    hcqt = np.stack(hcqt_list, axis=0) # (H, F, T)
    hcqt = np.transpose(hcqt, (0, 2, 1)) # (H, T, F)
    
    # Log-amplitude scaling (important for deep learning)
    hcqt = librosa.amplitude_to_db(hcqt, ref=np.max)
    
    # Normalize roughly to 0-1 or similar range (optional but good)
    # The paper often uses (hcqt + 80) / 80 approx for dB specs.
    hcqt = (hcqt + 80.0) / 80.0
    hcqt = np.clip(hcqt, 0.0, 1.0)

    return hcqt.astype(np.float32), freqs


def n_bins_from_octaves(n_octaves, bins_per_octave):
    return n_octaves * bins_per_octave

def load_f0_file(path):
    """Loads F0 file. Column 0: time, Column 2: F0 (Hz). 0 means unvoiced."""
    if not os.path.exists(path):
        return None, None
    data = np.loadtxt(path)
    times = data[:, 0]
    f0_hz = data[:, 2]
    return times, f0_hz

def f0_to_salience(f0_times, f0_hz, freqs_hz, frame_times, sigma_cents=25.0):
    """
    Converts F0 annotations to a time-frequency salience map with Gaussian blur.
    
    Args:
        f0_times: timestamps of the F0 annotations
        f0_hz: F0 values in Hz
        freqs_hz: center frequencies of the CQT bins
        frame_times: timestamps of the CQT frames
        sigma_cents: standard deviation of the Gaussian in cents
    
    Returns:
        salience: (T, F) matrix with values in [0, 1]
    """
    n_bins = len(freqs_hz)
    n_frames = len(frame_times)
    salience = np.zeros((n_frames, n_bins), dtype=np.float32)
    
    log2_freqs = np.log2(freqs_hz)
    
    # Interpolate F0 to frame times
    # Note: Ground truth is usually dense enough, but let's be safe.
    # MedleyDB pitch files are usually frames.
    f0_interp = np.interp(frame_times, f0_times, f0_hz, left=0.0, right=0.0)
    
    # Pre-compute constant factor for gaussian
    # Gaussian = exp(-0.5 * (diff_cents / sigma)^2)
    # diff_cents = 1200 * (log2(f) - log2(f0))
    
    for t, f0 in enumerate(f0_interp):
        if f0 <= 0:
            continue
            
        log2_f0 = np.log2(f0)
        
        # Vectorized gaussian for this frame
        diff_octaves = log2_freqs - log2_f0
        diff_cents = 1200.0 * diff_octaves
        
        # We only need to compute for bins reasonably close to f0 to save time
        # 3 sigma is usually sufficient. 3 * 25 = 75 cents = 0.75 semitones.
        # Bins per semitone = 3. So +/- 3 bins approx.
        # Let's do a wider window to be safe, e.g. +/- 50 cents.
        
        vals = np.exp(-0.5 * (diff_cents / sigma_cents) ** 2)
        
        # Peak normalize to 1.0 per frame (since it's a probability-like map)
        # Or simple exact gaussian values.
        # The assignment says "predict a pitch-related salience", usually binary cross entropy target is 0-1.
        # Max value of exp is 1.0 when diff is 0.
        
        # Threshold small values to keep it sparse/clean (optional)
        vals[vals < 1e-4] = 0
        
        salience[t, :] = vals

    return salience

class MedleyPitchDataset(Dataset):
    def __init__(self, data_dir, n_time_frames=50, split='train', split_ratio=0.8, seed=42):
        """
        Args:
            data_dir: Directory containing .wav and .f0.Corrected.txt files
            n_time_frames: Number of frames per input sample (for training context)
            split: 'train' or 'val'
            split_ratio: Ratio of files to use for training
        """
        self.data_dir = data_dir
        self.n_time_frames = n_time_frames
        
        # Find all paired wav and f0 files
        wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
        self.tracks = []
        
        for wav_path in wav_files:
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            # Try to find matching f0 file. The naming convention in the folder provided:
            # 01-D_AMairena.wav -> 01-D_AMairena.f0.Corrected.txt
            f0_path = os.path.join(data_dir, basename + ".f0.Corrected.txt")
            if os.path.exists(f0_path):
                self.tracks.append({'wav': wav_path, 'f0': f0_path, 'name': basename})
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(self.tracks)
        
        split_idx = int(len(self.tracks) * split_ratio)
        if split == 'train':
            self.tracks = self.tracks[:split_idx]
        else:
            self.tracks = self.tracks[split_idx:]
            
        if len(self.tracks) == 0:
            print(f"Warning: No tracks found for split {split} in {data_dir}")
            
        print(f"Split {split}: {len(self.tracks)} tracks.")

        # Cache loaded data to memory (Dataset is small)
        self.samples = [] # List of (hcqt_patch, salience_patch)
        self.prepare_data()

    def prepare_data(self):
        """
        Loads all tracks, computes HCQT and Salience, and chunks them into samples.
        """
        for track in self.tracks:
            print(f"Processing {track['name']}...")
            y, sr = librosa.load(track['wav'], sr=TARGET_SR, mono=True)
            
            # Compute HCQT
            hcqt, freqs = compute_hcqt(y, sr)
            # hcqt shape: (H, T, F)

            # Compute Salience
            f0_times, f0_hz = load_f0_file(track['f0'])
            n_frames = hcqt.shape[1]
            frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH)
            
            salience = f0_to_salience(f0_times, f0_hz, freqs, frame_times)
            # salience shape: (T, F)
            
            # Chunk into smaller examples
            n_chunks = n_frames // self.n_time_frames
            # We can use overlapping chunks or just disjoint. 
            # For training data augmentation, overlapping is good.
            # Here we do disjoint for simplicity, or simple stepping.
            
            step = self.n_time_frames // 2 # 50% overlap
            
            for i in range(0, n_frames - self.n_time_frames, step):
                hcqt_patch = hcqt[:, i:i+self.n_time_frames, :]
                salience_patch = salience[i:i+self.n_time_frames, :]
                
                # Check if patch has any voicing (optional? Avoid empty silent patches?)
                # If salience is all zeros, it's silence/unvoiced. Model should learn that too.
                # But maybe balance it if needed. For now, keep all.
                
                self.samples.append((hcqt_patch, salience_patch))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return tensors compatible with PyTorch Conv2d: (Channels, Time, Freq)
        # Currently HCQT is (H, T, F). This maps to (Channels, Height, Width) conceptually or (C, D1, D2).
        # PyTorch Conv2d expects (N, C, H, W). 
        # Here Time is one dimension, Freq is another.
        # Let's treat Time as Height, Freq as Width? Or vice versa.
        # Usually for spectrograms: (Batch, 1, Freq, Time) or (Batch, 1, Time, Freq).
        # Since we have 5 harmonics, we use them as Channels.
        # Output: (5, T, F)
        
        hcqt, salience = self.samples[idx]
        
        # Salience is (T, F). We need it to match model output shape.
        # Usually model output is (Batch, 1, T, F) or just (Batch, T, F).
        # Let's return (T, F).
        
        return torch.tensor(hcqt, dtype=torch.float32), torch.tensor(salience, dtype=torch.float32)

if __name__ == "__main__":
    # Test block
    data_dir = "trainData"
    if os.path.exists(data_dir):
        ds = MedleyPitchDataset(data_dir, split='train')
        print(f"Dataset size: {len(ds)}")
        if len(ds) > 0:
            x, y = ds[0]
            print(f"Sample shape: X={x.shape}, Y={y.shape}")
    else:
        print(f"Data directory {data_dir} not found. Please unzip Data.zip.")
