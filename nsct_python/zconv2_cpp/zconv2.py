import numpy as np

def _zconv2(x: np.ndarray, h: np.ndarray, mup: np.ndarray) -> np.ndarray:
    """
    2D convolution with upsampled filter using periodic boundary.
    Python translation of zconv2.c MEX file.
    
    This computes convolution as if the filter had been upsampled by matrix mup,
    but without actually upsampling the filter (efficient stepping through zeros).
    
    Args:
        x: Input signal (2D array)
        h: Filter (2D array)
        mup: Upsampling matrix (2x2 array) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    mup = np.array(mup, dtype=int)
    M0, M1, M2, M3 = mup[0, 0], mup[0, 1], mup[1, 0], mup[1, 1]
    
    s_row_len, s_col_len = x.shape
    f_row_len, f_col_len = h.shape
    
    # Calculate upsampled filter dimensions
    new_f_row_len = (M0 - 1) * (f_row_len - 1) + M2 * (f_col_len - 1) + f_row_len - 1
    new_f_col_len = (M3 - 1) * (f_col_len - 1) + M1 * (f_row_len - 1) + f_col_len - 1
    
    # Initialize output
    y = np.zeros_like(x)
    
    # Starting indices (center of upsampled filter)
    start1 = new_f_row_len // 2
    start2 = new_f_col_len // 2
    mn1 = start1 % s_row_len
    mn2 = mn2_save = start2 % s_col_len
    
    # Compute convolution
    for n1 in range(s_row_len):
        for n2 in range(s_col_len):
            out_index_x = mn1
            out_index_y = mn2
            sum_val = 0.0
            
            for l1 in range(f_row_len):
                index_x = out_index_x
                index_y = out_index_y
                
                for l2 in range(f_col_len):
                    sum_val += x[index_x, index_y] * h[l1, l2]
                    
                    # Step through input with M2, M3
                    index_x -= M2
                    if index_x < 0:
                        index_x += s_row_len
                    if index_x >= s_row_len:
                        index_x -= s_row_len
                        
                    index_y -= M3
                    if index_y < 0:
                        index_y += s_col_len
                
                # Step through for outer filter loop with M0, M1
                out_index_x -= M0
                if out_index_x < 0:
                    out_index_x += s_row_len
                    
                out_index_y -= M1
                if out_index_y < 0:
                    out_index_y += s_col_len
                if out_index_y >= s_col_len:
                    out_index_y -= s_col_len
            
            y[n1, n2] = sum_val
            
            mn2 += 1
            if mn2 >= s_col_len:
                mn2 -= s_col_len
        
        mn2 = mn2_save
        mn1 += 1
        if mn1 >= s_row_len:
            mn1 -= s_row_len
    
    return y
