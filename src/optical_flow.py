import cv2
import numpy as np
from tqdm import tqdm

def compute_optical_flow(video_path: str):
    """
    Computes dense optical flow (Farneback) for the given video.
    Returns:
        U: np.ndarray of shape (T-1, H, W) representing horizontal displacement.
        V: np.ndarray of shape (T-1, H, W) representing vertical displacement.
        fps: float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame from {video_path}")
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    U_list, V_list = [], []
    
    print(f"Extracting Optical Flow ({frame_count} frames)...")
    pbar = tqdm(total=frame_count-1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        U_list.append(flow[..., 0])
        V_list.append(flow[..., 1])
        
        prev_gray = gray
        pbar.update(1)
        
    pbar.close()
    cap.release()
    
    # Shape: (T, H, W)
    U = np.stack(U_list, axis=0)
    V = np.stack(V_list, axis=0)
    
    return U, V, fps
