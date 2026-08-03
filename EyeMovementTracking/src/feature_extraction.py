import numpy as np
import math
import logging

logger = logging.getLogger('EyeTracking')

def extract_features(history, fps=30.0):
    """
    Extract eye tracking features from history list of dicts.
    
    Each record in history:
    {
        'frame': idx,
        'left_raw': (x,y),
        'left_norm': (nx, ny),
        'right_raw': (x,y),
        'right_norm': (nx, ny),
        'left_ear': ear,
        'right_ear': ear,
        'left_pupil_radius': r,
        'right_pupil_radius': r
    }
    """
    n_frames = len(history)
    if n_frames < 5:
        return {
            'total_frames': n_frames,
            'blink_count': 0,
            'avg_blink_duration': 0.0,
            'blink_rate_per_min': 0.0,
            'fixation_count': 0,
            'avg_fixation_duration': 0.0,
            'saccade_count': 0,
            'avg_saccade_amplitude': 0.0,
            'pupil_area_std': 0.0,
            'saccadic_ratio': 0.0
        }
        
    # --- 1. Blink Extraction ---
    # We define a blink when EAR goes below 0.17 or pupil is lost (radius == 0)
    # We analyze the average EAR of left and right eyes
    ears = []
    for r in history:
        val = 0.0
        valid_ears = []
        if r['left_ear'] > 0: valid_ears.append(r['left_ear'])
        if r['right_ear'] > 0: valid_ears.append(r['right_ear'])
        
        # fallback: if pupil tracking lost, treat as potential blink
        if r['left_pupil_radius'] == 0 and r['right_pupil_radius'] == 0:
            val = 0.0
        elif len(valid_ears) > 0:
            val = np.mean(valid_ears)
        else:
            val = 0.3
        ears.append(val)
        
    # Find contiguous segments of low EAR
    blink_threshold = 0.18
    in_blink = False
    blink_durations = []
    current_blink_len = 0
    
    for val in ears:
        if val < blink_threshold:
            if not in_blink:
                in_blink = True
                current_blink_len = 1
            else:
                current_blink_len += 1
        else:
            if in_blink:
                in_blink = False
                # Filter out very short noise spikes (e.g. 1 frame)
                if current_blink_len >= 2:
                    blink_durations.append(current_blink_len)
                current_blink_len = 0
                
    # If file ends during a blink
    if in_blink and current_blink_len >= 2:
        blink_durations.append(current_blink_len)
        
    blink_count = len(blink_durations)
    avg_blink_duration = float(np.mean(blink_durations)) if blink_count > 0 else 0.0
    
    # Calculate blinks per minute
    duration_sec = n_frames / fps
    blink_rate_per_min = (blink_count / (duration_sec + 1e-6)) * 60.0
    
    # --- 2. Fixation and Saccade Extraction (I-VT Algorithm) ---
    # We use normalized gaze positions. We average left and right eyes for robustness.
    gazes = []
    for r in history:
        pts = []
        if r['left_norm'] is not None: pts.append(r['left_norm'])
        if r['right_norm'] is not None: pts.append(r['right_norm'])
        
        if len(pts) > 0:
            gazes.append(np.mean(pts, axis=0))
        else:
            gazes.append(None) # missing tracking
            
    # Calculate velocities (distance between consecutive frames)
    velocities = []
    for i in range(1, len(gazes)):
        g0 = gazes[i-1]
        g1 = gazes[i]
        
        if g0 is not None and g1 is not None:
            # Distance in normalized units
            v = np.linalg.norm(g1 - g0)
            velocities.append(v)
        else:
            velocities.append(None)
            
    # Classify frames using a velocity threshold (e.g., 0.03 normalized units per frame)
    # At 30 FPS, 0.03 normalized units corresponds to approx 100 pixels/sec on typical crops
    vel_threshold = 0.035
    
    # Label frames: 0 = fixation, 1 = saccade, None = missing
    labels = []
    for v in velocities:
        if v is None:
            labels.append(None)
        elif v > vel_threshold:
            labels.append(1) # Saccade
        else:
            labels.append(0) # Fixation
            
    # Group contiguous fixation and saccade segments
    fixations = []
    saccades = []
    
    current_segment = []
    current_label = None
    
    # Align labels with frames (length of labels is n_frames - 1)
    for idx, label in enumerate(labels):
        if label is None:
            # End current segment
            if len(current_segment) > 0:
                if current_label == 0:
                    fixations.append(current_segment)
                elif current_label == 1:
                    saccades.append(current_segment)
                current_segment = []
            current_label = None
            continue
            
        if current_label is None:
            current_label = label
            current_segment = [idx]
        elif current_label == label:
            current_segment.append(idx)
        else:
            # Label changed! Save current segment and start new one
            if current_label == 0:
                fixations.append(current_segment)
            elif current_label == 1:
                saccades.append(current_segment)
            current_label = label
            current_segment = [idx]
            
    # Save last segment
    if len(current_segment) > 0:
        if current_label == 0:
            fixations.append(current_segment)
        elif current_label == 1:
            saccades.append(current_segment)
            
    # Filter short segments
    # True fixations typically last at least 100ms (approx 3 frames at 30 fps)
    min_fixation_len = 3
    valid_fixations = [f for f in fixations if len(f) >= min_fixation_len]
    valid_saccades = saccades # Keep all saccades
    
    fixation_count = len(valid_fixations)
    avg_fixation_duration = float(np.mean([len(f) for f in valid_fixations])) if fixation_count > 0 else 0.0
    
    saccade_count = len(valid_saccades)
    
    # Saccadic Amplitude: distance between start and end coordinates of the saccade
    saccade_amplitudes = []
    for s in valid_saccades:
        start_idx = s[0]
        end_idx = s[-1] + 1
        if end_idx < len(gazes) and gazes[start_idx] is not None and gazes[end_idx] is not None:
            # Amplitude in pixels (assuming standard screen dimensions like 640x480 for plotting)
            amp = np.linalg.norm((gazes[end_idx] - gazes[start_idx]) * [640, 480])
            saccade_amplitudes.append(amp)
            
    avg_saccade_amplitude = float(np.mean(saccade_amplitudes)) if len(saccade_amplitudes) > 0 else 0.0
    
    # Saccadic Ratio (saccade frames / total frames)
    saccade_frames_count = sum([len(s) for s in valid_saccades])
    saccadic_ratio = float(saccade_frames_count / float(n_frames)) if n_frames > 0 else 0.0
    
    # --- 3. Pupil Size Variation ---
    # Standard deviation of pupil area (radius^2 * pi)
    pupil_radii = []
    for r in history:
        pr = []
        if r['left_pupil_radius'] > 0: pr.append(r['left_pupil_radius'])
        if r['right_pupil_radius'] > 0: pr.append(r['right_pupil_radius'])
        if len(pr) > 0:
            pupil_radii.append(np.mean(pr))
            
    if len(pupil_radii) > 2:
        pupil_areas = [math.pi * (r ** 2) for r in pupil_radii]
        pupil_area_std = float(np.std(pupil_areas))
    else:
        pupil_area_std = 0.0
        
    return {
        'total_frames': n_frames,
        'blink_count': blink_count,
        'avg_blink_duration': avg_blink_duration,
        'blink_rate_per_min': blink_rate_per_min,
        'fixation_count': fixation_count,
        'avg_fixation_duration': avg_fixation_duration,
        'saccade_count': saccade_count,
        'avg_saccade_amplitude': avg_saccade_amplitude,
        'pupil_area_std': pupil_area_std,
        'saccadic_ratio': saccadic_ratio
    }
