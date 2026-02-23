import os
import cv2
import numpy as np
from tqdm import tqdm
from optical_flow import compute_optical_flow
from storage import save_to_zarr

def process_video(
    input_path: str,
    output_dir: str,
    patch_size: int,
    freq_min: float,
    freq_max: float,
    saliency_threshold: float
):
    """
    The core pipeline:
    1. Extract Optical Flow U(t), V(t)
    2. Patch-based Spatial Aggregation
    3. Saliency Filtering (Variance over time)
    4. Complex-domain FFT over time
    5. Feature Extraction (Freq, Amp_U, Amp_V, Phase)
    6. Zarr output
    """
    # 1. Optical Flow
    U, V, fps = compute_optical_flow(input_path)
    T, H, W = U.shape
    
    print(f"Original Flow Shape: {U.shape} @ {fps} FPS")

    # 2. Patch-based Spatial Aggregation
    # Pad dimensions to be divisible by patch_size if necessary
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        U = np.pad(U, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
        V = np.pad(V, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
        
    _, H_pad, W_pad = U.shape
    new_H, new_W = H_pad // patch_size, W_pad // patch_size
    
    if patch_size > 1:
        # Reshape and mean over patches
        U = U.reshape(T, new_H, patch_size, new_W, patch_size).mean(axis=(2, 4))
        V = V.reshape(T, new_H, patch_size, new_W, patch_size).mean(axis=(2, 4))
        print(f"Patched Flow Shape: {U.shape}")
        
    # 3. Saliency Filtering & Aperture Problem Mitigation
    # The aperture problem causes optical flow along edges (like guitar strings) to be highly noisy.
    # To fix this, we project the flow vectors (u, v) onto the dominant spatial gradient direction.
    
    # Compute average spatial gradient over the video to find the true structural edges
    cap = cv2.VideoCapture(input_path)
    ret, first_frame = cap.read()
    if ret:
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Pad gradients to match U and V
        if pad_h > 0 or pad_w > 0:
            gx = np.pad(gx, ((0, pad_h), (0, pad_w)), mode='edge')
            gy = np.pad(gy, ((0, pad_h), (0, pad_w)), mode='edge')
            
        # Patch aggregation for gradients
        if patch_size > 1:
            gx = gx.reshape(new_H, patch_size, new_W, patch_size).mean(axis=(1, 3))
            gy = gy.reshape(new_H, patch_size, new_W, patch_size).mean(axis=(1, 3))
            
        # Normalize gradient vectors to get the normal direction of the edges
        grad_mag = np.sqrt(gx**2 + gy**2) + 1e-8
        nx = gx / grad_mag # Normal X component
        ny = gy / grad_mag # Normal Y component
        
        # Project Flow (U, V) onto the normal vector (nx, ny)
        # flow_proj = U * nx + V * ny 
        # Then the true corrected flow is flow_proj * nx and flow_proj * ny
        flow_proj = U * nx[np.newaxis, ...] + V * ny[np.newaxis, ...]
        U = flow_proj * nx[np.newaxis, ...]
        V = flow_proj * ny[np.newaxis, ...]
    cap.release()

    # Compute variance over time for U and V
    var_u = np.var(U, axis=0)
    var_v = np.var(V, axis=0)
    saliency_mask = (var_u + var_v) > saliency_threshold
    
    active_count = np.sum(saliency_mask)
    total_count = new_H * new_W
    print(f"Saliency Filter: Retained {active_count}/{total_count} patches "
          f"({active_count/total_count*100:.2f}%)")
    
    # Initialize output map (1, new_H, new_W, 4)
    # channels: [0] = Freq, [1] = Amp_U, [2] = Amp_V, [3] = Phase_U, [4] = Phase_V
    result_map = np.zeros((1, new_H, new_W, 5), dtype=np.float32)
    
    if active_count == 0:
        print("Warning: No patches passed saliency threshold. Saving empty output.")
    else:
        # Frequencies corresponding to FFT bins
        freqs = np.fft.fftfreq(T, d=1/fps)
        valid_freq_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
        
        if len(valid_freq_idx) == 0:
            print(f"Warning: No valid frequencies found in range [{freq_min}, {freq_max}]")
            # We will still output zeros but log warning.
            
        print("Performing Complex FFT...")
        
        # 4. Complex-domain FFT
        # Only process active patches to save time, or do it vectorized.
        # Given this is python, vectorized is faster if memory permits.
        # Let's vectorize over all active patches:
        
        U_active = U[:, saliency_mask] # Shape: (T, active_count)
        V_active = V[:, saliency_mask]
        
        # Z(t) = U(t) + i V(t)
        Z_active = U_active + 1j * V_active
        
        # FFT over time axis
        Z_fft = np.fft.fft(Z_active, axis=0) # Shape: (T, active_count)
        
        # Extract features
        if len(valid_freq_idx) > 0:
            # Mask out invalid freq bins by taking slice of valid ones
            Z_fft_valid = Z_fft[valid_freq_idx, :]
            freqs_valid = freqs[valid_freq_idx]
            
            # Find the peak frequency index for each active patch
            magnitudes = np.abs(Z_fft_valid) # Size (len(valid_idx), active_count)
            peak_idx = np.argmax(magnitudes, axis=0) # Size (active_count,)
            
            peak_freq_vals = freqs_valid[peak_idx] # Size (active_count,)
            # Get the complex value at the peak frequency
            peak_complex_vals = Z_fft_valid[peak_idx, np.arange(active_count)] 
            
            # Extract Amp_U, Amp_V, Phase
            # Z^*(f) has U(f) + i V(f), scaled by 2/T ideally 
            # Amp_U refers to magnitude in U component, Amp_V in V component
            # A common way is U_fft = Re(Z_fft) + i... wait.
            # U is real, V is real. 
            # FFT(u + iv) = FFT(u) + i FFT(v) because FFT is linear!
            # So Re(Z_fft) = FFT(u), Im(Z_fft) = FFT(v) is NOT correct if u,v are generic 
            # Wait, FFT of complex signal combines pos/neg frequencies.
            
            # A simpler feature representation: 
            # Dominant freq = peak_freq_vals
            # Magnitude = np.abs(peak_complex_vals) / T * 2
            # Phase = np.angle(peak_complex_vals)
            # Amp_u and Amp_v as projections of magnitude? 
            # Or directly calculating FFT(u), FFT(v) at that peak freq.
            
            U_fft_valid = np.fft.fft(U_active, axis=0)[valid_freq_idx, :]
            V_fft_valid = np.fft.fft(V_active, axis=0)[valid_freq_idx, :]
            
            peak_U = U_fft_valid[peak_idx, np.arange(active_count)]
            peak_V = V_fft_valid[peak_idx, np.arange(active_count)]
            
            amp_u = np.abs(peak_U) * 2 / T
            amp_v = np.abs(peak_V) * 2 / T
            phase_u = np.angle(peak_U)
            phase_v = np.angle(peak_V)
            
            # Fill result array shapes
            freq_arr = np.zeros(active_count, dtype=np.float32)
            ampu_arr = np.zeros(active_count, dtype=np.float32)
            ampv_arr = np.zeros(active_count, dtype=np.float32)
            phase_u_arr = np.zeros(active_count, dtype=np.float32)
            phase_v_arr = np.zeros(active_count, dtype=np.float32)
            
            freq_arr[:] = peak_freq_vals
            ampu_arr[:] = amp_u
            ampv_arr[:] = amp_v
            phase_u_arr[:] = phase_u
            phase_v_arr[:] = phase_v
            
            # Assign back to active patches
            res_view = result_map[0] # (new_H, new_W, 5)
            res_view[saliency_mask, 0] = freq_arr
            res_view[saliency_mask, 1] = ampu_arr
            res_view[saliency_mask, 2] = ampv_arr
            res_view[saliency_mask, 3] = phase_u_arr
            res_view[saliency_mask, 4] = phase_v_arr
            
    # 6. Saving results to Zarr
    base_name = os.path.basename(input_path)
    base_name_no_ext, _ = os.path.splitext(base_name)
    output_filename = f"{base_name_no_ext}.zarr"
    output_fullpath = os.path.join(output_dir, output_filename)
    
    meta = {
        'patch_size': int(patch_size),
        'fps': float(fps),
        'freq_min': float(freq_min),
        'freq_max': float(freq_max),
        'saliency_threshold': float(saliency_threshold)
    }
    
    save_to_zarr(output_fullpath, result_map, meta=meta)
