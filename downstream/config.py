
class SCQMedConfig():
    MODEL = "GRU"
    TASK = 'MIV'
    RATIO = 2/3

    SEED = 2023
    USE_CUDA = True
    GPU_ONLY = True
    GPU = '0'
    EPOCH = 1000
    DIM = 64
    LR = 5e-4 
    BATCH = 1024
    WD = 0
    DDI = 0.06
    KP = 0.08
    HIST = 3 

    ROOT = '../downstream_dataset/'
    LOG = './log/'

    ontology_ROOT = '../downstream_dataset/mimic-iv_data/ontology/'
    HIDVAE_DIAG_EMB = '../downstream_dataset/mimic-iv_data/mmrq_embedding/diag_fused_quantized_best.pt'
    HIDVAE_PROC_EMB = '../downstream_dataset/mimic-iv_data/mmrq_embedding/proc_fused_quantized_best.pt'
    HIDVAE_DRUG_EMB = '../downstream_dataset/mimic-iv_data/mmrq_embedding/med_fused_quantized_best.pt'
    HIDVAE_USE_SID = False
    HIDVAE_DIAG_SID = '../downstream_dataset/mimic-iv_data/mmrq_embedding/diag_fused_quantized_best.pt'
    HIDVAE_PROC_SID = '../downstream_dataset/mimic-iv_data/mmrq_embedding/proc_fused_quantized_best.pt'
    HIDVAE_DRUG_SID = '../downstream_dataset/mimic-iv_data/mmrq_embedding/med_fused_quantized_best.pt'


    SID_AGGREGATION = 'concat'

    PERF_TEST = False
    PERF_TEST_THRESHOLD = 0.5
    PERF_TEST_WARMUP_ROUNDS = 1

    FEWSHOT = False
    FEWSHOT_RATIO = 0.05 



config = vars(SCQMedConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
