# File path: configs/config.py

class Config:
    # FIX
    MODEL_MAX_LEN = 256
    BATCH_SIZE = 8
    MAX_EPOCHS = 10                   
    EARLY_STOPPING = True
    PATIENCE = 3                      
    MIN_DELTA = 1e-4 
    WEIGHT_DECAY = 0.01
    GRAD_CLIP_NORM = 1.0
    NUM_LABELS=5

    # TUNE
    TUNE_SEED = 42 # find best hyperparams with this seed then run on other seeds
    SEEDS = [42, 43, 44]  

    SELECT_METRIC = "micro_f1"  # "macro_f1"
    REPORT_METRICS = ["macro_f1", "micro_f1"]

    LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]
    DROPOUT_GRID = [0.1, 0.2]
    WARMUP_RATIO_GRID = [0.0, 0.1]
    
    # Default values if not tuning
    LR = 2e-5
    DROPOUT = 0.1
    WARMUP_RATIO = 0.1

    OUTPUT_DIR = "outputs"
    SAVE_BEST_ONLY = True
    LOG_EVERY_STEPS = 50