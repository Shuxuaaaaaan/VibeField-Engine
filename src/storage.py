import zarr
from numcodecs import Blosc
import os
import numpy as np

def save_to_zarr(output_path: str, data: np.ndarray, meta: dict = None):
    """
    Saves the extracted 4D tensor (T, H, W, C) into Zarr format.
    C usually contains 5 channels: [Dominant_Freq, Amp_u, Amp_v, Phase_u, Phase_v].
    """
    # Ensure intermediate directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    T, H, W, C = data.shape
    
    # Compress with Blosc LZ4
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)
    
    # Priority chunking on time dimension, e.g. large T, spatial as 64x64 blocks
    # Here T is 1 as we are computing FFT over the entire time window.
    chunk_H = min(H, 64)
    chunk_W = min(W, 64)
    chunks = (1, chunk_H, chunk_W, C)
    
    root = zarr.open(
        output_path, 
        mode='w', 
        shape=(T, H, W, C), 
        chunks=chunks, 
        dtype=data.dtype,
        zarr_format=2,
        compressor=compressor
    )
    
    root[:] = data
    if meta is not None:
        root.attrs.update(meta)
    print(f"Successfully saved results to {output_path} (Shape: {root.shape})")
