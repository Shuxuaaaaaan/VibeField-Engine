import cv2
import numpy as np
import zarr
import argparse
import sys
import os

def get_color_map():
    # Pre-calculate a color lut (0 to 255)
    # Blue to Red colormap (120 Hue to 0 Hue)
    color_lut = []
    for i in range(256):
        h = (1.0 - (i / 255.0)) * 120
        hsv = np.uint8([[[h, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
        color_lut.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return color_lut

# Global variables for trackbars
g_scale_val = 100
g_speed_val = 100
g_thresh_min_val = 2
g_thresh_max_val = 100

def on_trackbar(val):
    pass

def play_video_with_vectors(video_path: str, zarr_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        sys.exit(1)
        
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr file {zarr_path} not found.")
        sys.exit(1)

    print(f"Loading Zarr data from {zarr_path}...")
    root = zarr.open(zarr_path, mode='r')
    data = root[:]  # Shape (1, H', W', 5)
    
    if data.shape[-1] != 5:
        print(f"Error: Expected 5 channels in Zarr data, but got {data.shape[-1]}.")
        print("Please re-run the engine with the updated code.")
        sys.exit(1)
        
    _, new_H, new_W, _ = data.shape
    
    res_view = data[0]
    freqs = res_view[..., 0]      # (H', W')
    amp_u = res_view[..., 1]      # (H', W')
    amp_v = res_view[..., 2]      # (H', W')
    phase_u = res_view[..., 3]    # (H', W')
    phase_v = res_view[..., 4]    # (H', W')
    
    # The engine sets active patches, and empty ones remain 0
    active_mask = freqs > 0
    
    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    patch_size_h = orig_H // new_H
    patch_size_w = orig_W // new_W
    patch_size = min(patch_size_h, patch_size_w)
    
    if patch_size == 0:
        patch_size = 1
        
    print(f"Video {orig_W}x{orig_H} @ {fps} FPS")
    print(f"Estimated Patch Size: {patch_size}")
    
    y_idxs, x_idxs = np.where(active_mask)
    centers_x = (x_idxs * patch_size + patch_size // 2).astype(float)
    centers_y = (y_idxs * patch_size + patch_size // 2).astype(float)
    
    act_freqs = freqs[active_mask]
    act_amp_u = amp_u[active_mask]
    act_amp_v = amp_v[active_mask]
    act_phase_u = phase_u[active_mask]
    act_phase_v = phase_v[active_mask]
    
    amps = np.sqrt(act_amp_u**2 + act_amp_v**2)
    if len(amps) == 0:
        print("No active patches found.")
        sys.exit(0)
        
    p98_amp = np.percentile(amps, 98)
    if p98_amp == 0:
        p98_amp = 1.0
        
    # Set the initial arrow scale so max vectors overlap 2 patches
    init_arrow_scale = (patch_size * 2.0) / p98_amp
    
    color_lut = get_color_map()
    
    frame_idx = 0
    dt = 1.0 / fps
    
    print("Starting Viewer... ")
    print("Use the trackbars in the Control Panel window to adjust settings.")
    print("Press 'q' or ESC in the image window to quit.")
    
    cv2.namedWindow("VibeField-Engine Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VibeField-Engine Viewer", orig_W * 2, orig_H * 2)

    # Create a separate window for controls to avoid cluttering the video
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 200)

    # Trackbars
    # Scale: 1 to 500 (representing 0.01x to 5.0x of init_arrow_scale, default 100 = 1.0x)
    cv2.createTrackbar("Vector Scale (%)", "Controls", 100, 500, on_trackbar)
    
    # Speed: 1 to 300 (representing 0.01x to 3.0x speed, default 100 = 1.0x)
    cv2.createTrackbar("Play Speed (%)", "Controls", 100, 300, on_trackbar)
    
    # Min Threshold: 0 to 100 (representing 0% to 100% of P98 amplitude, default 2%)
    cv2.createTrackbar("Min Thresh (%)", "Controls", 2, 100, on_trackbar)
    
    # Max Threshold: 1 to 200 (representing 1% to 200% of P98 amplitude, default 100%)
    cv2.createTrackbar("Max Thresh (%)", "Controls", 100, 200, on_trackbar)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue
            
        t = frame_idx * dt
        
        # Read trackbars
        scale_perc = cv2.getTrackbarPos("Vector Scale (%)", "Controls") / 100.0
        speed_perc = cv2.getTrackbarPos("Play Speed (%)", "Controls") / 100.0
        min_th_perc = cv2.getTrackbarPos("Min Thresh (%)", "Controls") / 100.0
        max_th_perc = cv2.getTrackbarPos("Max Thresh (%)", "Controls") / 100.0
        
        # Prevent zero scale / speed
        if scale_perc == 0: scale_perc = 0.01
        if speed_perc == 0: speed_perc = 0.01
        if max_th_perc < min_th_perc: max_th_perc = min_th_perc + 0.01
        
        current_arrow_scale = init_arrow_scale * scale_perc
        
        # Calculate instantaneous displacement vectors
        omega = 2 * np.pi * act_freqs * t
        
        inst_u = act_amp_u * np.cos(omega + act_phase_u)
        inst_v = act_amp_v * np.cos(omega + act_phase_v)
        
        draw_u = inst_u * current_arrow_scale
        draw_v = inst_v * current_arrow_scale
        inst_mags = np.sqrt(inst_u**2 + inst_v**2)
        
        # Filter based on thresholds relative to P98 amplitude
        draw_mask = (inst_mags >= (p98_amp * min_th_perc)) & (inst_mags <= (p98_amp * max_th_perc))
        
        d_cx = centers_x[draw_mask]
        d_cy = centers_y[draw_mask]
        d_du = draw_u[draw_mask]
        d_dv = draw_v[draw_mask]
        d_mags = inst_mags[draw_mask]
        
        # Dynamic coloring
        intensities = np.clip(d_mags / p98_amp, 0.0, 1.0)
        color_idxs = (intensities * 255).astype(int)
        
        for i in range(len(d_cx)):
            cx = d_cx[i]
            cy = d_cy[i]
            du = d_du[i]
            dv = d_dv[i]
            
            cx_dyn = int(cx + du)
            cy_dyn = int(cy + dv)
            
            end_x = int(cx_dyn + du * 1.5)
            end_y = int(cy_dyn + dv * 1.5)
            
            c_idx = color_idxs[i]
            color = color_lut[c_idx]
            
            cv2.arrowedLine(frame, (cx_dyn, cy_dyn), (end_x, end_y), color, 1, tipLength=0.3)
            cv2.circle(frame, (cx_dyn, cy_dyn), 1, color, -1)
                
        text = f"Time: {t:.2f}s | Frame: {frame_idx} | Scale: x{current_arrow_scale:.1f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    
        cv2.imshow("VibeField-Engine Viewer", frame)
        
        # Adjust playback delay based on speed
        base_delay = 1000 / fps
        adjusted_delay = int(base_delay / speed_perc)
        if adjusted_delay < 1: adjusted_delay = 1
        
        key = cv2.waitKey(adjusted_delay) & 0xFF
        
        if key == 27 or key == ord('q'): 
            break
            
        frame_idx += 1
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="VibeField-Engine GUI Viewer")
    parser.add_argument("-i", "--input", required=True, help="Path to original input video")
    parser.add_argument("-z", "--zarr", required=True, help="Path to the generated Zarr file")
    
    args = parser.parse_args()
    play_video_with_vectors(args.input, args.zarr)

if __name__ == "__main__":
    main()
