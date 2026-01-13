"""
Experiment scripts for fingerprint recognition evaluation.

This package contains experiment scripts for evaluating different
fingerprint recognition methods:

1. exp_ssim_baseline.py - Image-based baseline evaluation
2. exp_minutiae_matching.py - Classical minutiae matching
3. exp_mcc_descriptor.py - MCC descriptor matching
4. exp_cnn_embedding.py - CNN-based embedding models
5. exp_cross_sensor.py - Cross-sensor evaluation

Running Experiments:
-------------------
From the project root:

    python experiments/exp_ssim_baseline.py --data_dir data/raw/FVC2004/DB1_B
    python experiments/exp_minutiae_matching.py --data_dir data/raw/FVC2004/DB1_B
    python experiments/exp_mcc_descriptor.py --compare
    python experiments/exp_cnn_embedding.py --train --epochs 50
    python experiments/exp_cross_sensor.py --data_dir data/raw
"""
