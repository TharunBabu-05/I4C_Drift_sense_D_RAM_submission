"""
Phase-2 Configuration Module
=============================
Defines hyperparameter defaults and search space settings for Phase-2 inference.
"""

class Phase2Config:
    # 1. Scale Search Grid
    COARSE_SCALES = [8.0, 9.0, 10.0, 11.0, 12.0]
    FINE_SCALE_RADIUS = 0.6
    FINE_SCALE_STEPS = [0.25, 0.1]
    
    # 2. Rotation Search Grid (degrees, CCW positive)
    COARSE_THETAS = [-5.0, -2.5, 0.0, 2.5, 5.0]
    FINE_THETA_RADIUS = 2.0
    FINE_THETA_STEPS = [1.0, 0.5]
    
    # 3. Hybrid Score Fusion Weights (NCC weight alpha, Siamese weight 1 - alpha)
    NCC_WEIGHT = 0.3
    SIAMESE_WEIGHT = 0.7
    
    # 4. Absent-Target Rejection Threshold
    REJECTION_THRESHOLD = 0.42
    
    # 5. Candidate Selection & Decoy Handling
    TOP_K_COARSE = 10
    CENTER_BIAS_WEIGHT = 0.05
    
    # 6. Confidence Calibration
    CONFIDENCE_SLOPE = 12.0
    
    # 7. Subpixel Refinement Window
    SUBPIXEL_RADIUS = 3
