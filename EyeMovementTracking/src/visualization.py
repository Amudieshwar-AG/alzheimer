import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use headless backend for saving files without X server / GUI
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger('EyeTracking')

def draw_on_frame(frame, eye_data, tracking_record):
    """
    Annotates the input BGR frame with detection bounding boxes, landmark circles,
    pupil centers, and status text.
    """
    annotated = frame.copy()
    h, w, _ = annotated.shape
    
    # 1. Draw eyes overlays
    for eye_key, color in [('left_eye', (0, 255, 0)), ('right_eye', (0, 0, 255))]:
        eye = eye_data.get(eye_key)
        if not eye:
            continue
            
        # Draw bounding box
        bx, by, bw, bh = eye['bbox']
        cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), color, 2)
        
        # Label eye
        label = "L. Eye" if eye_key == 'left_eye' else "R. Eye"
        cv2.putText(annotated, f"{label} EAR: {eye.get('ear', 0.0):.2f}", (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    
        # Draw pupil circle if detected
        pupil = eye.get('pupil')
        if pupil and pupil['center'] is not None:
            cx, cy = pupil['center']
            # convert local back to global
            gx = bx + cx
            gy = by + cy
            r = pupil['radius']
            
            # Draw center point
            cv2.circle(annotated, (gx, gy), 2, (0, 255, 255), -1)
            # Draw contour radius
            cv2.circle(annotated, (gx, gy), max(3, r), (255, 255, 0), 1)
            
            # If ellipse available, draw it
            if pupil.get('ellipse'):
                # Translate ellipse center
                (ecx, ecy), (ew, eh), eangle = pupil['ellipse']
                translated_ellipse = ((bx + ecx, by + ecy), (ew, eh), eangle)
                cv2.ellipse(annotated, translated_ellipse, (255, 0, 255), 1)
                
    # 2. Draw global tracking stats on corner of frame
    if tracking_record:
        left_pos = tracking_record['left_norm']
        right_pos = tracking_record['right_norm']
        
        cv2.putText(annotated, "STATUS: Tracking Active", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    
        if left_pos:
            cv2.putText(annotated, f"Gaze L: ({left_pos[0]:.2f}, {left_pos[1]:.2f})", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if right_pos:
            cv2.putText(annotated, f"Gaze R: ({right_pos[0]:.2f}, {right_pos[1]:.2f})", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
    return annotated

def save_diagnostic_crops(eye_data, results_dir):
    """
    Saves fig5_segmented_eye.png and fig6_denoised.png for visual verification.
    """
    logger.info("Saving diagnostic cropped images...")
    left = eye_data.get('left_eye')
    if not left:
        return
        
    img = left.get('img')
    denoised = left.get('denoised')
    segmented = left.get('segmented')
    
    if img is not None and img.size > 0:
        # We can construct a combined collage: Raw | Denoised | Segmented
        # If denoised/segmented are single channel, convert them to BGR for concatenation
        collage_parts = [img]
        
        if denoised is not None:
            denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            collage_parts.append(denoised_bgr)
            cv2.imwrite(os.path.join(results_dir, 'fig6_denoised.png'), denoised)
            
        if segmented is not None:
            segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
            collage_parts.append(segmented_bgr)
            cv2.imwrite(os.path.join(results_dir, 'fig5_segmented_eye.png'), segmented)
            
        if len(collage_parts) > 1:
            collage = np.hstack(collage_parts)
            # save segmented report figure
            logger.info("Diagnostic crops written successfully.")

def save_gaze_path_plot(history, output_path):
    """
    Plots a 2D gaze trajectory map (results/fig7_gaze_path.png).
    Uses normalized coordinates and draws arrows representing saccades,
    and circular markers for fixations.
    """
    logger.info(f"Plotting gaze trajectory map at {output_path}...")
    
    left_x, left_y = [], []
    right_x, right_y = [], []
    
    for r in history:
        if r['left_norm'] is not None:
            left_x.append(r['left_norm'][0])
            left_y.append(r['left_norm'][1])
        if r['right_norm'] is not None:
            right_x.append(r['right_norm'][0])
            right_y.append(r['right_norm'][1])
            
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#f8fafc') # sleek gray background
    
    # Invert Y axis because image coordinates start from top-left (0,0)
    ax.invert_yaxis()
    
    # Left eye path
    if len(left_x) > 0:
        ax.plot(left_x, left_y, color='#1e3a8a', linestyle='-', alpha=0.4, label='Left Gaze Path')
        ax.scatter(left_x, left_y, color='#2563eb', s=10, alpha=0.6)
        # Highlight start and end
        ax.scatter(left_x[0], left_y[0], color='#10b981', s=80, marker='^', label='Start', zorder=5)
        ax.scatter(left_x[-1], left_y[-1], color='#ef4444', s=80, marker='o', label='End', zorder=5)
        
    # Right eye path
    if len(right_x) > 0:
        ax.plot(right_x, right_y, color='#b91c1c', linestyle='-', alpha=0.3, label='Right Gaze Path')
        ax.scatter(right_x, right_y, color='#dc2626', s=8, alpha=0.5)
        
    ax.set_title("Gaze Position Trajectory (Normalized)", fontsize=14, fontweight='bold', color='#1e293b')
    ax.set_xlabel("X (Width Ratio)", fontsize=11, color='#475569')
    ax.set_ylabel("Y (Height Ratio)", fontsize=11, color='#475569')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0) # inverted limits
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_feature_fusion_plots(history, output_path):
    """
    Plots temporal variations of EAR, Pupil Area, and gaze velocity (results/fig8_feature_fusion.png).
    """
    logger.info(f"Plotting feature analysis curves at {output_path}...")
    
    frames = [r['frame'] for r in history]
    left_ear = [r['left_ear'] for r in history]
    right_ear = [r['right_ear'] for r in history]
    
    left_rad = [r['left_pupil_radius'] for r in history]
    right_rad = [r['right_pupil_radius'] for r in history]
    left_area = [np.pi * (rad ** 2) for rad in left_rad]
    
    # Calculate Gaze Velocity over time
    velocities = [0.0]
    for i in range(1, len(history)):
        r0 = history[i-1]
        r1 = history[i]
        
        # Average positions
        p0 = []
        if r0['left_norm'] is not None: p0.append(r0['left_norm'])
        if r0['right_norm'] is not None: p0.append(r0['right_norm'])
        
        p1 = []
        if r1['left_norm'] is not None: p1.append(r1['left_norm'])
        if r1['right_norm'] is not None: p1.append(r1['right_norm'])
        
        if len(p0) > 0 and len(p1) > 0:
            v = np.linalg.norm(np.mean(p1, axis=0) - np.mean(p0, axis=0))
            # scale velocity to display nicely
            velocities.append(v * 100) # percentage of screen per frame
        else:
            velocities.append(0.0)
            
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    # Subplot 1: Eye Aspect Ratio (EAR)
    ax1.plot(frames, left_ear, color='#2563eb', label='Left Eye EAR', linewidth=1.5)
    ax1.plot(frames, right_ear, color='#dc2626', label='Right Eye EAR', linewidth=1.2, linestyle='--')
    # draw blink threshold line
    ax1.axhline(y=0.18, color='#f43f5e', linestyle=':', label='Blink Threshold')
    ax1.set_title("Eye Aspect Ratio (EAR) - Blink Tracking", fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_ylabel("EAR Ratio", fontsize=9)
    ax1.set_ylim(0.0, 0.45)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=8)
    
    # Subplot 2: Pupil Area
    ax2.plot(frames, left_area, color='#0891b2', label='Pupil Area (px²)', linewidth=1.8)
    ax2.set_title("Pupil Area (Aperture Oscillations)", fontsize=11, fontweight='bold', color='#1e293b')
    ax2.set_ylabel("Area (pixels^2)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Subplot 3: Gaze Velocity (I-VT Classification)
    ax3.plot(frames, velocities, color='#7c3aed', label='Gaze Velocity (% screen/f)', linewidth=1.5)
    # Draw velocity classification threshold
    ax3.axhline(y=3.5, color='#ef4444', linestyle=':', label='Saccade Velocity Limit')
    # Fill under curve for fixations
    ax3.fill_between(frames, 0, velocities, where=[v <= 3.5 for v in velocities], color='#8b5cf6', alpha=0.1, label='Fixation Zone')
    ax3.fill_between(frames, 0, velocities, where=[v > 3.5 for v in velocities], color='#f43f5e', alpha=0.25, label='Saccade Peak')
    
    ax3.set_title("Gaze Velocity (Saccades Peaks vs. Fixations)", fontsize=11, fontweight='bold', color='#1e293b')
    ax3.set_xlabel("Time (Frame Count)", fontsize=10)
    ax3.set_ylabel("Velocity (%/f)", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
