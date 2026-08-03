import numpy as np
import logging

logger = logging.getLogger('EyeTracking')

def fuse_features(extracted_features):
    """
    Fuses individual temporal features to compute advanced biomarkers.
    
    Returns a dict with fused indices:
    - cognitive_load_index: related to pupil variation and blink rates.
    - gaze_stability_index: related to fixations duration.
    - saccadic_exploration_rate: ratio of saccades to fixations.
    - composite_cognitive_risk: clinical risk index based on weighted biomarkers (0.0 to 1.0)
    - feature_vector: flat list of key biomarkers for ML classification.
    """
    feat = extracted_features
    
    # 1. Saccadic Exploration Rate (saccades per fixation)
    fix_count = feat.get('fixation_count', 0)
    sac_count = feat.get('saccade_count', 0)
    exploration_rate = float(sac_count) / float(fix_count + 1)
    
    # 2. Gaze Stability Index (overall fixation persistence)
    # Normed relative to processed frames (100 frame baseline)
    avg_fix_dur = feat.get('avg_fixation_duration', 0.0)
    gaze_stability = (fix_count * avg_fix_dur) / 100.0
    
    # 3. Cognitive Load Index (Hippus fluctuation vs. blink suppression)
    pupil_std = feat.get('pupil_area_std', 0.0)
    blink_rate = feat.get('blink_rate_per_min', 0.0)
    cognitive_load = pupil_std / (blink_rate / 60.0 + 1.0) # normalized by blink rate per sec
    
    # 4. Composite Risk Assessment Score (mock biomarker index for cognitive decline)
    # Risk parameters:
    # - Low gaze stability (shorter, scattered fixations) -> higher risk
    # - High exploration rate (hyperactive saccades/drift) -> higher risk
    # - High pupil variability (inability to focus) -> higher risk
    # - Abnormally low/high blink rate
    
    # Normalize features into [0, 1] range weights
    w_stability = max(0.0, min(1.0, 1.0 - (gaze_stability / 5.0))) # stable = low risk
    w_exploration = max(0.0, min(1.0, exploration_rate / 3.0))      # high rate = higher risk
    w_pupil = max(0.0, min(1.0, pupil_std / 500.0))                 # high variance = higher risk
    
    risk_score = 0.4 * w_stability + 0.3 * w_exploration + 0.3 * w_pupil
    risk_score = max(0.0, min(1.0, risk_score))
    
    # 5. Build flat feature vector for machine learning classifiers
    feature_vector = [
        float(feat.get('blink_count', 0)),
        float(feat.get('avg_blink_duration', 0.0)),
        float(feat.get('blink_rate_per_min', 0.0)),
        float(fix_count),
        float(avg_fix_dur),
        float(sac_count),
        float(feat.get('avg_saccade_amplitude', 0.0)),
        float(pupil_std),
        float(feat.get('saccadic_ratio', 0.0)),
        exploration_rate,
        gaze_stability,
        cognitive_load,
        risk_score
    ]
    
    fused = {
        'saccadic_exploration_rate': exploration_rate,
        'gaze_stability_index': gaze_stability,
        'cognitive_load_index': cognitive_load,
        'composite_risk_score': risk_score,
        'feature_vector': feature_vector,
        'vector_labels': [
            'blink_count', 'avg_blink_duration', 'blink_rate_per_min',
            'fixation_count', 'avg_fixation_duration', 'saccade_count',
            'avg_saccade_amplitude', 'pupil_area_std', 'saccadic_ratio',
            'exploration_rate', 'gaze_stability_index', 'cognitive_load_index',
            'composite_risk_score'
        ]
    }
    
    logger.info(f"Feature fusion completed. Composite Risk Score: {risk_score:.2f}")
    return fused
