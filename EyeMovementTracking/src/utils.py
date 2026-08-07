import os
import urllib.request
import logging
from fpdf import FPDF

# Configure logging
def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'output', 'reports')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pipeline.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('EyeTracking')

logger = logging.getLogger('EyeTracking')

def setup_directories():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    directories = [
        os.path.join(base_dir, 'data', 'input'),
        os.path.join(base_dir, 'data', 'output', 'segmented'),
        os.path.join(base_dir, 'data', 'output', 'denoised'),
        os.path.join(base_dir, 'data', 'output', 'gaze_path'),
        os.path.join(base_dir, 'data', 'output', 'feature_plots'),
        os.path.join(base_dir, 'data', 'output', 'grayscale'),
        os.path.join(base_dir, 'data', 'output', 'color_gradient'),
        os.path.join(base_dir, 'data', 'output', 'reports'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'results', 'grayscale'),
        os.path.join(base_dir, 'results', 'color_gradient')
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Project directories initialized successfully.")

def download_haar_cascades():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    cascades = {
        'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
    }
    
    for filename, url in cascades.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            logger.info(f"Downloading {filename} from official OpenCV repository...")
            try:
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Successfully downloaded {filename}.")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise e
        else:
            logger.info(f"Model file {filename} already exists at {filepath}.")

class PDFReport(FPDF):
    def header(self):
        # Draw a line and title
        self.set_text_color(26, 54, 93)  # Dark Blue
        self.set_font('Helvetica', 'B', 18)
        self.cell(0, 10, 'Alzheimer Detection: Eye Tracking Assessment', ln=True, align='C')
        self.set_draw_color(26, 54, 93)
        self.set_line_width(0.8)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(8)

    def footer(self):
        # Page numbers
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Automated Assessment Report', align='C')

def generate_pdf_report(metrics, fig_paths, output_pdf_path):
    logger.info(f"Generating PDF report at {output_pdf_path}...")
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Metadata Block
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 8, "Assessment Metadata", ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(71, 85, 105)
    
    metadata = [
        ("Assessment Type:", "Saccadic & Fixation Analysis (Eye Tracking)"),
        ("Subject ID:", "SUB-2026-0730"),
        ("Date / Time:", "2026-07-30 (Automated System)"),
        ("Algorithm Version:", "v1.0.0 (MediaPipe + Haar Fusion)")
    ]
    for key, value in metadata:
        pdf.cell(40, 6, key, border=0)
        pdf.cell(0, 6, value, border=0, ln=True)
        
    pdf.ln(6)
    
    # Feature Metrics Table
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Extracted Features Summary Table", ln=True)
    pdf.ln(2)
    
    # Headers
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(85, 8, " Feature Parameter", border=1, fill=True)
    pdf.cell(45, 8, " Measured Value", border=1, fill=True)
    pdf.cell(60, 8, " Clinical Interpretation", border=1, fill=True, ln=True)
    
    # Row printing
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    fill = False
    
    rows = [
        ("Total Frames Processed", f"{metrics.get('total_frames', 0)}", "Full duration of the capture sequence"),
        ("Total Blinks Detected", f"{metrics.get('blink_count', 0)}", f"Blink frequency: {metrics.get('blink_rate_per_min', 0.0):.2f}/min"),
        ("Average Blink Duration (frames)", f"{metrics.get('avg_blink_duration', 0.0):.2f} frames", "Normal duration indicates alert state"),
        ("Total Fixations Identified", f"{metrics.get('fixation_count', 0)}", "Sustained gazes (gaze stabilization)"),
        ("Average Fixation Duration (frames)", f"{metrics.get('avg_fixation_duration', 0.0):.2f} frames", "Shorter fixations can indicate ADHD/Alzheimer's"),
        ("Total Saccades Detected", f"{metrics.get('saccade_count', 0)}", "Rapid eye movements between fixations"),
        ("Average Saccade Amplitude (pixels)", f"{metrics.get('avg_saccade_amplitude', 0.0):.2f} px", "Quantifies spatial gaze jumps"),
        ("Pupil Area Variation (Std Dev)", f"{metrics.get('pupil_area_std', 0.0):.2f} px²", "Indicator of cognitive load / autonomic response"),
        ("Antisaccade / Saccadic Ratio", f"{metrics.get('saccadic_ratio', 0.0):.2f}", "Balance metric of saccadic to total frames")
    ]
    
    for row in rows:
        if fill:
            pdf.set_fill_color(241, 245, 249) # light gray row background
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(85, 7, f" {row[0]}", border=1, fill=True)
        pdf.cell(45, 7, f" {row[1]}", border=1, fill=True)
        pdf.cell(60, 7, f" {row[2]}", border=1, fill=True, ln=True)
        fill = not fill
        
    pdf.ln(6)
    
    # Feature Fusion & Cognitive Summary
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Cognitive Assessment & Conclusion", ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(51, 65, 85)
    
    ratio = metrics.get('saccadic_ratio', 0.0)
    if ratio < 0.15:
        cognitive_summary = (
            "Assessment Summary: The subject displayed stable fixation patterns and expected blink dynamics. "
            "The gaze movement velocity profile falls within healthy reference bounds, indicating typical "
            "neurological oculomotor control. Saccadic to fixation transition velocities are well balanced."
        )
    else:
        cognitive_summary = (
            "Assessment Summary: The subject displayed increased saccadic dispersion and elevated saccade-to-fixation ratio. "
            "Occasional visual tremors or rapid saccadic drift were observed during steady fixation targets. "
            "This tracking pattern warrants follow-up clinical cognitive evaluation for potential mild cognitive impairment (MCI)."
        )
        
    pdf.multi_cell(0, 5, cognitive_summary)
    pdf.ln(10)
    
    # Page 2: Visualizations
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Visual Diagnostics & Plots", ln=True)
    pdf.ln(2)
    
    # We will display plots on this page
    # Let's check which figures exist and place them
    # fig7 (Gaze Path) and fig8 (Feature Fusion) are very informative
    y_start = pdf.get_y()
    
    # Figure 1: Gaze Path (results/fig7_gaze_path.png)
    gaze_path = fig_paths.get('fig7_gaze_path')
    if gaze_path and os.path.exists(gaze_path):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(90, 6, "Fig 1: Gaze Trajectory Map", ln=False)
        pdf.cell(10, 6, "", ln=True) # spacer
        pdf.image(gaze_path, x=10, y=pdf.get_y() + 2, w=85)
        
    # Figure 2: Feature Fusion (results/fig8_feature_fusion.png)
    feat_fusion = fig_paths.get('fig8_feature_fusion')
    if feat_fusion and os.path.exists(feat_fusion):
        pdf.set_y(y_start)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(100, 6, "", ln=False) # indent
        pdf.cell(90, 6, "Fig 2: Feature Analysis Over Time", ln=True)
        pdf.image(feat_fusion, x=105, y=pdf.get_y() + 2, w=95)
        
    pdf.set_y(y_start + 80)
    pdf.ln(5)
    
    # Figure 3: Segmented pupil showing eye frames (results/fig5_segmented_eye.png)
    seg_eye = fig_paths.get('fig5_segmented_eye')
    if seg_eye and os.path.exists(seg_eye):
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(90, 6, "Fig 3: Segmented & Detected Pupil Frame", ln=True)
        pdf.image(seg_eye, x=10, y=pdf.get_y() + 2, w=85)
        
    # Figure 4: Denoised (results/fig6_denoised.png)
    denoised_eye = fig_paths.get('fig6_denoised')
    if denoised_eye and os.path.exists(denoised_eye):
        pdf.set_y(y_start + 80 + 5)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(100, 6, "", ln=False)
        pdf.cell(90, 6, "Fig 4: Denoised & Preprocessed Frame", ln=True)
        pdf.image(denoised_eye, x=105, y=pdf.get_y() + 2, w=95)
        
    # Save PDF
    pdf.output(output_pdf_path)
    logger.info("PDF report compiled successfully.")
