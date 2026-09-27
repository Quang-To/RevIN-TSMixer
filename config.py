SCENARIO   = 2          
MODEL      = "nbeats"  
VAL_METRIC = "pinball"  
SEED       = 42
DATA_FILE  = "data_simulated_s1.csv"  # hoặc "data_simulated_s2.csv", "data_TSI_v2.csv"

SEQ_LENGTH = 24
PRED_LEN   = 1         

# TSMixer
N_BLOCK    = 2
FF_DIM     = 64

# NBEATS
N_STACKS   = 2
NBEATS_N_BLOCKS = 3
N_LAYERS   = 3
LAYER_DIM  = 384

# NHITS
N_BLOCKS   = 2
HIDDEN_DIM = 256

DROPOUT    = 0.069

BATCH_SIZE = 16
LR         = 1.95e-4
EPOCHS     = 3000
PATIENCE   = 100

HOLDING_COST     = 4.0      
SHORTAGE_COST    = 30.0      
                           
ORDERING_COST    = 50_000.0 
LEAD_TIME        = 1         
FORECAST_HORIZON = 1

USE_LOG_RETURN = False
