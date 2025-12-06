import torch
import numpy as np
import librosa
import mir_eval
import os
import glob
import argparse
from train_model import DeepSalience
from data_set_prep import compute_hcqt, load_f0_file, TARGET_SR, HOP_LENGTH, FMIN, BINS_PER_OCTAVE, N_OCTAVES

def predict_on_file(model, wav_path, device='cpu'):
    # Load audio
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    
    # Compute HCQT: (H, T, F)
    # We need to reshape for model: (1, H, T, F) - Batch size 1
    # But wait, audio can be long. We might run out of GPU memory.
    # We can perform inference on the whole spectrogram if it fits, or chunk it.
    # For this assignment, tracks are ~30s-2min. 
    # (5, 2000, 360) floats is not too big (~14MB). It should fit on most GPUs.
    
    hcqt, freqs = compute_hcqt(y, sr)
    hcqt_tensor = torch.tensor(hcqt, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        # returns logits (1, 1, T, F)
        logits = model(hcqt_tensor)
        salience = torch.sigmoid(logits)
        
    salience = salience.squeeze().cpu().numpy() # (T, F)
    
    return salience, freqs

def salience_to_f0(salience, freqs, times, threshold=0.5):
    """
    Decodes salience map to F0 curve using peak picking.
    """
    f0_est = []
    
    for t in range(salience.shape[0]):
        # Simple frame-wise peak picking
        frame_sal = salience[t, :]
        max_idx = np.argmax(frame_sal)
        max_val = frame_sal[max_idx]
        
        if max_val >= threshold:
            f0 = freqs[max_idx]
            f0_est.append(f0)
        else:
            f0_est.append(0.0) # Unvoiced
            
    return np.array(f0_est)

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = DeepSalience().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model file {args.model_path} not found. Using random weights (bad!).")
    
    # Find files (Validation set logic or all)
    # We mimic the split logic to find validation files if requested
    # Find files (Validation set logic or all)
    # We mimic the split logic to find validation files if requested
    # Support new directory structure: audio/*.wav and pitch/*.csv
    
    search_path = os.path.join(args.data_dir, "**", "*.wav")
    wav_files = sorted(glob.glob(search_path, recursive=True))
    paired_files = []
    
    for wav_path in wav_files:
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Strategy: look for 'pitch' directory parallel to 'audio' dir
        parent_dir = os.path.dirname(wav_path) # .../audio
        if os.path.basename(parent_dir) == 'audio':
            grandparent = os.path.dirname(parent_dir)
            pitch_dir = os.path.join(grandparent, 'pitch')
            f0_path = os.path.join(pitch_dir, basename + ".csv")
        else:
            # Fallback
            f0_path = os.path.join(parent_dir, basename + ".csv")
            
        if os.path.exists(f0_path):
            paired_files.append((wav_path, f0_path))
            
    # Deterministic split to match training
    import random
    random.seed(42)
    random.shuffle(paired_files)
    
    split_idx = int(len(paired_files) * 0.8)
    if args.split == 'val':
        eval_files = paired_files[split_idx:]
        print(f"Evaluating on Validation split ({len(eval_files)} files)")
    elif args.split == 'train':
        eval_files = paired_files[:split_idx]
        print(f"Evaluating on Training split ({len(eval_files)} files)")
    else:
        eval_files = paired_files
        print(f"Evaluating on ALL files ({len(eval_files)} files)")

    if len(eval_files) == 0:
        print("No files to evaluate.")
        return

    # Metrics storage
    all_metrics = []
    
    for wav_path, ref_path in eval_files:
        print(f"Processing {os.path.basename(wav_path)}...")
        
        # 1. Prediction
        salience, freqs = predict_on_file(model, wav_path, device)
        n_frames = salience.shape[0]
        est_times = librosa.frames_to_time(np.arange(n_frames), sr=TARGET_SR, hop_length=HOP_LENGTH)
        
        # 2. Decoding
        est_freqs = salience_to_f0(salience, freqs, est_times, threshold=args.threshold)
        
        # 3. Reference
        ref_times, ref_freqs = load_f0_file(ref_path)
        
        # 4. Evaluation using mir_eval
        # mir_eval expects matching time grids or resampled.
        # usually we resample estimate to reference times, or vice versa.
        # mir_eval.melody.evaluate handles this if we provide (ref_time, ref_freq) and (est_time, est_freq)
        
        # Note: mir_eval might complain if time bases are too different or offsets.
        # usually simplest is to resample est to ref times.
        est_freqs_resampled = np.interp(ref_times, est_times, est_freqs, left=0.0, right=0.0)
        
        # mir_eval expects numpy arrays
        # ref_freqs has 0 for unvoiced. mir_eval handles positive Hz + negative/0 indication or separate voicing array.
        # Actually mir_eval.melody.evaluate takes (ref_time, ref_freq) and (est_time, est_freq).
        # Frequencies <= 0 are unvoiced.
        
        metrics = mir_eval.melody.evaluate(ref_times, ref_freqs, ref_times, est_freqs_resampled)
        all_metrics.append(metrics)
        
        # Print individual file breakdown
        print(f"  RPA: {metrics['Raw Pitch Accuracy']:.3f}, VR: {metrics['Voicing Recall']:.3f}, FA: {metrics['Voicing False Alarm']:.3f}")

    # Aggregate
    if all_metrics:
        avg_metrics = {}
        for k in all_metrics[0].keys():
            avg_metrics[k] = np.mean([m[k] for m in all_metrics])
            
        print("\n=== Overall Results ===")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="trainData")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--split", type=str, default="val", help="val, train, or all")
    parser.add_argument("--threshold", type=float, default=0.3, help="Salience threshold for peak picking")
    args = parser.parse_args()
    
    evaluate(args)
