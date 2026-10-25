"""Shared constants for training (Kaggle) and local inference.

The label layout follows the RSNA 2024 Lumbar Spine Degenerative
Classification competition: 5 conditions x 5 levels = 25 targets, each
graded Normal/Mild, Moderate or Severe.
"""

LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
LEVEL_LABELS = {"l1_l2": "L1-L2", "l2_l3": "L2-L3", "l3_l4": "L3-L4", "l4_l5": "L4-L5", "l5_s1": "L5-S1"}

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]
CONDITION_LABELS_TR = {
    "spinal_canal_stenosis": "Spinal kanal darlığı",
    "left_neural_foraminal_narrowing": "Sol nöral foraminal darlık",
    "right_neural_foraminal_narrowing": "Sağ nöral foraminal darlık",
    "left_subarticular_stenosis": "Sol subartiküler (lateral reses) darlık",
    "right_subarticular_stenosis": "Sağ subartiküler (lateral reses) darlık",
}

# Target order: condition-major, same as the RSNA train.csv columns.
TARGETS = [f"{c}_{l}" for c in CONDITIONS for l in LEVELS]

SEVERITIES = ["Normal/Mild", "Moderate", "Severe"]
SEVERITY_LABELS_TR = ["Normal/Hafif", "Orta", "Ağır"]
SEVERITY_WEIGHTS = [1.0, 2.0, 4.0]  # competition metric weights

# Input series ("roles") and which condition is best seen on which series.
SERIES_ROLES = ["sag_t1", "sag_t2", "ax_t2"]
SERIES_ROLE_LABELS = {"sag_t1": "Sagittal T1", "sag_t2": "Sagittal T2/STIR", "ax_t2": "Axial T2"}
RSNA_DESCRIPTION_TO_ROLE = {"Sagittal T1": "sag_t1", "Sagittal T2/STIR": "sag_t2", "Axial T2": "ax_t2"}
CONDITION_TO_ROLE = {
    "spinal_canal_stenosis": "sag_t2",
    "left_neural_foraminal_narrowing": "sag_t1",
    "right_neural_foraminal_narrowing": "sag_t1",
    "left_subarticular_stenosis": "ax_t2",
    "right_subarticular_stenosis": "ax_t2",
}

SLICES_PER_SERIES = {"sag_t1": 10, "sag_t2": 10, "ax_t2": 10}
IMAGE_SIZE = 256
