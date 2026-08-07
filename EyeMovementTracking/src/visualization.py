import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use headless backend for saving files without X server / GUI
import matplotlib.pyplot as plt
import os
import logging
from src.segmentation import generate_edge_segmentation

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
        output_seg_dir = os.path.join(os.path.dirname(results_dir), 'data', 'output', 'segmented')
        os.makedirs(output_seg_dir, exist_ok=True)
        
        if denoised is not None:
            cv2.imwrite(os.path.join(results_dir, 'fig6_denoised.png'), denoised)
            
        edge_seg = generate_edge_segmentation(img)
        if edge_seg is not None:
            cv2.imwrite(os.path.join(results_dir, 'fig5_segmented_eye.png'), edge_seg)
            cv2.imwrite(os.path.join(output_seg_dir, 'fig5_segmented_eye.png'), edge_seg)
        elif segmented is not None:
            cv2.imwrite(os.path.join(results_dir, 'fig5_segmented_eye.png'), segmented)
            cv2.imwrite(os.path.join(output_seg_dir, 'fig5_segmented_eye.png'), segmented)
            
        logger.info("Diagnostic crops written successfully.")
            
        base_dir = os.path.dirname(results_dir)
        save_grayscale_and_gradient_images(eye_data, None, base_dir)

def compute_color_gradient(img, colormap=cv2.COLORMAP_JET):
    """
    Computes standard color gradient heatmap from the original input image.
    """
    if img is None or img.size == 0:
        return None
        
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    color_gradient = cv2.applyColorMap(gray, colormap)
    return color_gradient

def enhance_grayscale(img):
    """
    Converts the original input image to standard clean grayscale.
    """
    if img is None or img.size == 0:
        return None
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    return gray

def save_grayscale_and_gradient_images(eye_data, frame, base_dir, timestamp=None):
    """
    Saves clean original grayscale images to output/grayscale and color gradient visualizations to output/color_gradient.
    """
    logger.info("Saving grayscale and color gradient images...")
    
    results_dir = os.path.join(base_dir, 'results')
    output_gray_dir = os.path.join(base_dir, 'data', 'output', 'grayscale')
    output_grad_dir = os.path.join(base_dir, 'data', 'output', 'color_gradient')
    results_gray_dir = os.path.join(results_dir, 'grayscale')
    results_grad_dir = os.path.join(results_dir, 'color_gradient')
    
    os.makedirs(output_gray_dir, exist_ok=True)
    os.makedirs(output_grad_dir, exist_ok=True)
    os.makedirs(results_gray_dir, exist_ok=True)
    os.makedirs(results_grad_dir, exist_ok=True)
    
    ts_suffix = f"_{timestamp}" if timestamp else ""
    
    # 1. Full Frame Grayscale & Color Gradient
    if frame is not None and frame.size > 0:
        gray_frame = enhance_grayscale(frame)
        grad_frame = compute_color_gradient(frame, cv2.COLORMAP_JET)
        
        cv2.imwrite(os.path.join(results_gray_dir, 'full_frame_grayscale.png'), gray_frame)
        cv2.imwrite(os.path.join(output_gray_dir, f'full_frame_grayscale{ts_suffix}.png'), gray_frame)
        
        if grad_frame is not None:
            cv2.imwrite(os.path.join(results_grad_dir, 'full_frame_color_gradient.png'), grad_frame)
            cv2.imwrite(os.path.join(output_grad_dir, f'full_frame_color_gradient{ts_suffix}.png'), grad_frame)
            
    # 2. Eye Crops Grayscale & Color Gradient (Only save if distinct from full frame)
    if eye_data:
        for eye_key in ['left_eye', 'right_eye']:
            eye = eye_data.get(eye_key)
            if not eye:
                continue
            
            raw_img = eye.get('img')
            denoised = eye.get('denoised')
            
            source_img = raw_img if raw_img is not None else denoised
            if source_img is not None and source_img.size > 0:
                # If crop shape matches full frame shape, skip duplicate left/right eye files
                if frame is not None and source_img.shape == frame.shape and np.array_equal(source_img, frame):
                    continue
                    
                gray_eye = enhance_grayscale(source_img)
                grad_eye = compute_color_gradient(source_img, cv2.COLORMAP_JET)
                
                cv2.imwrite(os.path.join(results_gray_dir, f'{eye_key}_grayscale.png'), gray_eye)
                cv2.imwrite(os.path.join(output_gray_dir, f'{eye_key}_grayscale{ts_suffix}.png'), gray_eye)
                
                if grad_eye is not None:
                    cv2.imwrite(os.path.join(results_grad_dir, f'{eye_key}_color_gradient.png'), grad_eye)
                    cv2.imwrite(os.path.join(output_grad_dir, f'{eye_key}_color_gradient{ts_suffix}.png'), grad_eye)


from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go

def save_gaze_path_plot(history, output_path, fps=30.0):
    """
    Plots:
    1. Primary 2D Paper Figure: output_path (results/fig7_gaze_path.png)
    2. Supplementary 3D Figures:
       - fig7_gaze_path_3d.png (Angle 1: elev=25, azim=-60)
       - fig7_gaze_path_3d_angle2.png (Angle 2: elev=15, azim=30)
       - fig7_gaze_path_3d.html (Interactive Plotly 3D)
    """
    logger.info(f"Plotting primary 2D and supplementary 3D gaze path plots at {output_path}...")
    
    results_dir = os.path.dirname(output_path)
    
    left_x, left_y, left_t = [], [], []
    right_x, right_y, right_t = [], [], []
    
    for r in history:
        f = r['frame']
        t_sec = f / fps if fps > 0 else f / 30.0
        if r['left_norm'] is not None:
            left_x.append(r['left_norm'][0])
            left_y.append(r['left_norm'][1])
            left_t.append(t_sec)
        if r['right_norm'] is not None:
            right_x.append(r['right_norm'][0])
            right_y.append(r['right_norm'][1])
            right_t.append(t_sec)

    # ----------------------------------------------------
    # 1. PRIMARY 2D PAPER FIGURE (fig7_gaze_path.png)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#f8fafc')
    ax.invert_yaxis()
    
    if len(left_x) > 0:
        if len(left_x) > 1:
            ax.plot(left_x, left_y, color='#3266ad', linestyle='-', linewidth=1.5, alpha=0.7, label='Left Gaze Path')
        ax.scatter(left_x, left_y, color='#3266ad', s=15, alpha=0.6)
        ax.scatter([left_x[0]], [left_y[0]], color='#10b981', s=80, marker='^', label='Start Fixation', zorder=5)
        if len(left_x) > 1:
            ax.scatter([left_x[-1]], [left_y[-1]], color='#d95926', s=80, marker='o', label='End Target', zorder=5)
            
    if len(right_x) > 0:
        if len(right_x) > 1:
            ax.plot(right_x, right_y, color='#d95926', linestyle='-', linewidth=1.5, alpha=0.7, label='Right Gaze Path')
        ax.scatter(right_x, right_y, color='#d95926', s=15, alpha=0.6)

    if len(left_x) == 0 and len(right_x) == 0:
        ax.scatter([0.5], [0.5], color='#6366f1', s=100, marker='X', label='Center Gaze Prior (0.5, 0.5)', zorder=5)
        
    ax.set_title("Gaze Position Trajectory (Normalized)", fontsize=13, fontweight='bold', color='#1e293b')
    ax.set_xlabel("X (Width Ratio)", fontsize=10, color='#475569')
    ax.set_ylabel("Y (Height Ratio)", fontsize=10, color='#475569')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    # ----------------------------------------------------
    # 2. SUPPLEMENTARY 3D MATPLOTLIB FIGURES (fig7_gaze_path_3d.png)
    # ----------------------------------------------------
    # Gaussian smooth x and y per-eye for 3D visualization (sigma=1.5), do NOT smooth t
    sx_l = gaussian_filter1d(left_x, sigma=1.5) if len(left_x) >= 3 else np.array(left_x)
    sy_l = gaussian_filter1d(left_y, sigma=1.5) if len(left_y) >= 3 else np.array(left_y)
    st_l = np.array(left_t)
    
    sx_r = gaussian_filter1d(right_x, sigma=1.5) if len(right_x) >= 3 else np.array(right_x)
    sy_r = gaussian_filter1d(right_y, sigma=1.5) if len(right_y) >= 3 else np.array(right_y)
    st_r = np.array(right_t)
    
    fig3d = plt.figure(figsize=(9, 7))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.set_facecolor('#ffffff')
    
    if len(sx_l) > 0:
        if len(sx_l) > 1:
            ax3d.plot(sx_l, sy_l, st_l, color='#3266ad', lw=1.5, label='Left Gaze Path')
        ax3d.scatter(sx_l, sy_l, st_l, color='#3266ad', s=15, alpha=0.6)
        ax3d.scatter([sx_l[0]], [sy_l[0]], [st_l[0]], color='#10b981', s=80, marker='^', label='Start Fixation', depthshade=False)
        if len(sx_l) > 1:
            ax3d.scatter([sx_l[-1]], [sy_l[-1]], [st_l[-1]], color='#d95926', s=80, marker='o', label='End Target', depthshade=False)

    if len(sx_r) > 0:
        if len(sx_r) > 1:
            ax3d.plot(sx_r, sy_r, st_r, color='#d95926', lw=1.5, label='Right Gaze Path')
        ax3d.scatter(sx_r, sy_r, st_r, color='#d95926', s=15, alpha=0.6)
        
    if len(sx_l) == 0 and len(sx_r) == 0:
        ax3d.scatter([0.5], [0.5], [0.0], color='#6366f1', s=100, marker='X', label='Center Gaze Prior', depthshade=False)

    ax3d.set_xlabel('X (Width Ratio)', fontsize=9)
    ax3d.set_ylabel('Y (Height Ratio)', fontsize=9)
    ax3d.set_zlabel('Time (s)', fontsize=9)
    ax3d.set_title('Gaze Position Trajectory (Normalized) — 3D', fontsize=12, fontweight='bold')
    ax3d.set_xlim(0.0, 1.0)
    ax3d.set_ylim(1.0, 0.0)
    ax3d.legend(loc='upper left', fontsize=8)
    
    # Save Angle 1 (elev=25, azim=-60)
    path_3d_p1 = os.path.join(results_dir, 'fig7_gaze_path_3d.png')
    ax3d.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(path_3d_p1, dpi=200)
    
    # Save Angle 2 (elev=15, azim=30)
    path_3d_p2 = os.path.join(results_dir, 'fig7_gaze_path_3d_angle2.png')
    ax3d.view_init(elev=15, azim=30)
    plt.savefig(path_3d_p2, dpi=200)
    plt.close()

    # ----------------------------------------------------
    # 3. INTERACTIVE PLOTLY 3D FIGURE (fig7_gaze_path_3d.html)
    # ----------------------------------------------------
    try:
        plotly_fig = go.Figure()
        
        if len(sx_l) > 0:
            plotly_fig.add_trace(go.Scatter3d(
                x=sx_l, y=sy_l, z=st_l,
                mode='lines+markers',
                line=dict(color='#3266ad', width=4),
                marker=dict(size=3, color='#3266ad', opacity=0.7),
                name='Left Gaze Path'
            ))
            plotly_fig.add_trace(go.Scatter3d(
                x=[sx_l[0]], y=[sy_l[0]], z=[st_l[0]],
                mode='markers',
                marker=dict(size=8, color='#10b981', symbol='diamond'),
                name='Start Fixation'
            ))
            if len(sx_l) > 1:
                plotly_fig.add_trace(go.Scatter3d(
                    x=[sx_l[-1]], y=[sy_l[-1]], z=[st_l[-1]],
                    mode='markers',
                    marker=dict(size=8, color='#d95926', symbol='circle'),
                    name='End Target'
                ))
                
        if len(sx_r) > 0:
            plotly_fig.add_trace(go.Scatter3d(
                x=sx_r, y=sy_r, z=st_r,
                mode='lines+markers',
                line=dict(color='#d95926', width=4),
                marker=dict(size=3, color='#d95926', opacity=0.7),
                name='Right Gaze Path'
            ))

        plotly_fig.update_layout(
            title='Gaze Position Trajectory (Normalized) — Interactive 3D',
            scene=dict(
                xaxis_title='X (Width Ratio)',
                yaxis_title='Y (Height Ratio)',
                zaxis_title='Time (s)',
                yaxis=dict(autorange='reversed')
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        html_out = os.path.join(results_dir, 'fig7_gaze_path_3d.html')
        plotly_fig.write_html(html_out)
        
        try:
            png_plotly_out = os.path.join(results_dir, 'fig7_gaze_path_3d_plotly.png')
            plotly_fig.write_image(png_plotly_out, width=1000, height=800, scale=2)
        except Exception as ke:
            logger.debug(f"Kaleido static export notice: {ke}")
    except Exception as pe:
        logger.warning(f"Could not generate Plotly 3D HTML for gaze path: {pe}")


def save_feature_fusion_plots(history, output_path):
    """
    Saves:
    1. Primary 2D Paper Figure: output_path (results/fig8_feature_fusion.png)
       - 3 2D subplots (EAR, Pupil Area, Gaze Velocity) matching original 2D format.
    2. Supplementary 3D Figures:
       - fig8_feature_fusion_3d.png (3D Subplots for EAR ribbon with 0.18 threshold plane, un-smoothed Pupil Area ribbon at Y=2, and Velocity ribbon at Y=3 with 3.5%/f plane & orange vertical saccade peak drop-lines!)
       - fig8_feature_fusion_3d.html (Interactive Plotly 3D)
    """
    logger.info(f"Plotting primary 2D and supplementary 3D feature fusion plots at {output_path}...")
    
    results_dir = os.path.dirname(output_path)
    
    frames = [r['frame'] for r in history]
    left_ear = [r['left_ear'] for r in history]
    right_ear = [r['right_ear'] for r in history]
    
    left_rad = [r['left_pupil_radius'] for r in history]
    left_area = [np.pi * (rad ** 2) for rad in left_rad]
    
    # Calculate Gaze Velocity over time
    velocities = [0.0]
    for i in range(1, len(history)):
        r0 = history[i-1]
        r1 = history[i]
        
        p0 = []
        if r0['left_norm'] is not None: p0.append(r0['left_norm'])
        if r0['right_norm'] is not None: p0.append(r0['right_norm'])
        
        p1 = []
        if r1['left_norm'] is not None: p1.append(r1['left_norm'])
        if r1['right_norm'] is not None: p1.append(r1['right_norm'])
        
        if len(p0) > 0 and len(p1) > 0:
            v = np.linalg.norm(np.mean(p1, axis=0) - np.mean(p0, axis=0))
            velocities.append(v * 100)
        else:
            velocities.append(0.0)

    is_single_frame = len(frames) == 1
    
    # ----------------------------------------------------
    # 1. PRIMARY 2D PAPER FIGURE (fig8_feature_fusion.png)
    # ----------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    # Subplot 1: Eye Aspect Ratio (EAR)
    if not is_single_frame:
        ax1.plot(frames, left_ear, color='#2563eb', label='Left Eye EAR', linewidth=1.5)
        ax1.plot(frames, right_ear, color='#dc2626', label='Right Eye EAR', linewidth=1.2, linestyle='--')
    ax1.scatter(frames, left_ear, color='#2563eb', s=35, zorder=5)
    ax1.scatter(frames, right_ear, color='#dc2626', s=25, zorder=5)
    ax1.axhline(y=0.18, color='#f43f5e', linestyle=':', label='Blink Threshold')
    ax1.set_title("Eye Aspect Ratio (EAR) - Blink Tracking", fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_ylabel("EAR Ratio", fontsize=9)
    ax1.set_ylim(0.0, 0.45)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=8)
    
    # Subplot 2: Pupil Area (Raw, un-smoothed)
    if not is_single_frame:
        ax2.plot(frames, left_area, color='#0891b2', label='Pupil Area (px²)', linewidth=1.8)
    ax2.scatter(frames, left_area, color='#0891b2', s=35, zorder=5, label='Measured Pupil Area')
    ax2.set_title("Pupil Area (Aperture Oscillations)", fontsize=11, fontweight='bold', color='#1e293b')
    ax2.set_ylabel("Area (px²)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)
    
    # Subplot 3: Gaze Velocity (I-VT Classification)
    if not is_single_frame:
        ax3.plot(frames, velocities, color='#7c3aed', label='Gaze Velocity (% screen/f)', linewidth=1.5)
        ax3.fill_between(frames, 0, velocities, where=[v <= 3.5 for v in velocities], color='#8b5cf6', alpha=0.1, label='Fixation Zone')
        ax3.fill_between(frames, 0, velocities, where=[v > 3.5 for v in velocities], color='#f43f5e', alpha=0.25, label='Saccade Peak')
    ax3.scatter(frames, velocities, color='#7c3aed', s=35, zorder=5)
    ax3.axhline(y=3.5, color='#ef4444', linestyle=':', label='Saccade Velocity Limit')
    
    ax3.set_title("Gaze Velocity (Saccades Peaks vs. Fixations)", fontsize=11, fontweight='bold', color='#1e293b')
    ax3.set_xlabel("Time (Frame Count)", fontsize=10)
    ax3.set_ylabel("Velocity (%/f)", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)
    
    if is_single_frame:
        ax3.set_xlim(0, 2)
        ax3.set_xticks([0, 1, 2])
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    # ----------------------------------------------------
    # 2. SUPPLEMENTARY 3D MATPLOTLIB FIGURES (fig8_feature_fusion_3d.png)
    # Combined 3D Figure with 3 Subplots (Option 1 - Separate Honest Z units)
    # ----------------------------------------------------
    fig3d = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Figure B - EAR Ribbons (X=Time, Y=Channel 0/1, Z=EAR)
    ax3d1 = fig3d.add_subplot(131, projection='3d')
    ax3d1.plot(frames, [0]*len(frames), left_ear, color='#2563eb', lw=2.0, label='Left Eye EAR')
    ax3d1.plot(frames, [1]*len(frames), right_ear, color='#dc2626', lw=2.0, linestyle='--', label='Right Eye EAR')
    
    # 0.18 Blink Threshold Plane
    if len(frames) > 0:
        min_f, max_f = min(frames), max(frames)
        if min_f == max_f:
            max_f = min_f + 1
        t_grid, y_grid = np.meshgrid([min_f, max_f], [-0.2, 1.2])
        z_thresh = np.full_like(t_grid, 0.18)
        ax3d1.plot_surface(t_grid, y_grid, z_thresh, color='red', alpha=0.15)
        
    ax3d1.set_yticks([0, 1])
    ax3d1.set_yticklabels(['Left Eye', 'Right Eye'], fontsize=8)
    ax3d1.set_xlabel('Time (Frame Count)', fontsize=8)
    ax3d1.set_zlabel('EAR Ratio', fontsize=8)
    ax3d1.set_title('Figure B: 3D EAR Ribbons & Blink Plane', fontsize=10, fontweight='bold')
    ax3d1.set_zlim(0.0, 0.45)
    ax3d1.legend(loc='upper right', fontsize=7)
    
    # Subplot 2: Figure C - Pupil Area Ribbon (X=Time, Y=Fixed 2, Z=Pupil Area)
    # Raw values, no smoothing to preserve detection plateaus/spikes
    ax3d2 = fig3d.add_subplot(132, projection='3d')
    ax3d2.plot(frames, [2]*len(frames), left_area, color='#199e70', lw=1.5, label='Pupil Area')
    ax3d2.scatter(frames, [2]*len(frames), left_area, color='#199e70', s=8, label='Measured Pupil Area')
    ax3d2.set_yticks([2])
    ax3d2.set_yticklabels(['Pupil Channel'], fontsize=8)
    ax3d2.set_xlabel('Time (Frame Count)', fontsize=8)
    ax3d2.set_zlabel('Pupil Area (px²)', fontsize=8)
    ax3d2.set_title('Figure C: 3D Pupil Area Ribbon (Raw)', fontsize=10, fontweight='bold')
    ax3d2.legend(loc='upper right', fontsize=7)
    
    # Subplot 3: Figure D - Gaze Velocity Ribbon (X=Time, Y=Fixed 3, Z=Velocity)
    ax3d3 = fig3d.add_subplot(133, projection='3d')
    ax3d3.plot(frames, [3]*len(frames), velocities, color='#7f77dd', lw=1.5, label='Gaze Velocity')
    
    # 3.5%/f Saccade Threshold Plane
    if len(frames) > 0:
        min_f, max_f = min(frames), max(frames)
        if min_f == max_f:
            max_f = min_f + 1
        t_grid, y_grid = np.meshgrid([min_f, max_f], [2.8, 3.2])
        z_vthresh = np.full_like(t_grid, 3.5)
        ax3d3.plot_surface(t_grid, y_grid, z_vthresh, color='orange', alpha=0.15)
        
    # Vertical orange drop lines for saccade peaks (velocity > 3.5)
    for i in range(len(frames)):
        if velocities[i] > 3.5:
            ax3d3.plot([frames[i], frames[i]], [3, 3], [0, velocities[i]], color='orange', lw=1.0, alpha=0.7, label='Saccade Peak' if i==0 else '')
            
    ax3d3.set_yticks([3])
    ax3d3.set_yticklabels(['Velocity Channel'], fontsize=8)
    ax3d3.set_xlabel('Time (Frame Count)', fontsize=8)
    ax3d3.set_zlabel('Velocity (%/f)', fontsize=8)
    ax3d3.set_title('Figure D: 3D Gaze Velocity & Saccade Peaks', fontsize=10, fontweight='bold')
    ax3d3.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    feat_3d_p1 = os.path.join(results_dir, 'fig8_feature_fusion_3d.png')
    plt.savefig(feat_3d_p1, dpi=200)
    plt.close()

    # ----------------------------------------------------
    # 3. INTERACTIVE PLOTLY 3D FIGURE (fig8_feature_fusion_3d.html)
    # ----------------------------------------------------
    try:
        pfig8 = go.Figure()
        
        # EAR Traces
        pfig8.add_trace(go.Scatter3d(
            x=frames, y=[0]*len(frames), z=left_ear,
            mode='lines+markers',
            line=dict(color='#2563eb', width=4),
            marker=dict(size=3, color='#2563eb'),
            name='Left Eye EAR'
        ))
        pfig8.add_trace(go.Scatter3d(
            x=frames, y=[1]*len(frames), z=right_ear,
            mode='lines+markers',
            line=dict(color='#dc2626', width=3),
            marker=dict(size=3, color='#dc2626'),
            name='Right Eye EAR'
        ))
        
        # Pupil Area Trace
        pfig8.add_trace(go.Scatter3d(
            x=frames, y=[2]*len(frames), z=left_area,
            mode='lines+markers',
            line=dict(color='#199e70', width=4),
            marker=dict(size=4, color='#199e70'),
            name='Pupil Area (px²)'
        ))
        
        # Velocity Trace
        pfig8.add_trace(go.Scatter3d(
            x=frames, y=[3]*len(frames), z=velocities,
            mode='lines+markers',
            line=dict(color='#7f77dd', width=4),
            marker=dict(size=4, color='#7f77dd'),
            name='Gaze Velocity (%/f)'
        ))
        
        pfig8.update_layout(
            title='Feature Fusion 3D Channel Analysis (EAR, Pupil Area, Velocity)',
            scene=dict(
                xaxis_title='Time (Frame Count)',
                yaxis_title='Channel (0:Left EAR, 1:Right EAR, 2:Pupil Area, 3:Velocity)',
                zaxis_title='Signal Value',
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        html_out8 = os.path.join(results_dir, 'fig8_feature_fusion_3d.html')
        pfig8.write_html(html_out8)
        
        try:
            png_pout8 = os.path.join(results_dir, 'fig8_feature_fusion_3d_plotly.png')
            pfig8.write_image(png_pout8, width=1000, height=800, scale=2)
        except Exception as ke:
            logger.debug(f"Kaleido static export notice for fig8: {ke}")
    except Exception as pe:
        logger.warning(f"Could not generate Plotly 3D HTML for feature fusion: {pe}")
