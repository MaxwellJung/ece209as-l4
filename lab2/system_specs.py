# Input shape is 3x32x32
# Network Definition
# Format: Conv [#F, #C, X, Y] -> number of filters, number of channels, filter width, filter height
# 1. Conv [32,3,3,3]
# 2. Conv [64,32,3,3]
# 3. Maxpool2d (reduce both width and height of activation by 2, to 16x16)
# 4. Conv [128,64,3,3]
# 5. Conv [256, 128, 3,3]
# 6. Conv [512, 256, 3, 3]

conv_layers = [
    {'input_width':32, 'input_height':32,
     'num_filter':32, 'num_channel':3, 'filter_width':3, 'filter_height':3},
    {'input_width':32, 'input_height':32,
     'num_filter':64, 'num_channel':32, 'filter_width':3, 'filter_height':3},
    {'input_width':16, 'input_height':16,
     'num_filter':128, 'num_channel':64, 'filter_width':3, 'filter_height':3},
    {'input_width':16, 'input_height':16,
     'num_filter':256, 'num_channel':128, 'filter_width':3, 'filter_height':3},
    {'input_width':16, 'input_height':16,
     'num_filter':512, 'num_channel':256, 'filter_width':3, 'filter_height':3},
]

# System Specs
ACTIVATION_BITWIDTH = 8
BITS_PER_INPUT = ACTIVATION_BITWIDTH
BITS_PER_OUTPUT = ACTIVATION_BITWIDTH
WEIGHT_BITWIDTH = 8
BITS_PER_WEIGHT = WEIGHT_BITWIDTH
DRAM_ACC_ENERGY_PER_BIT = 4e-12
SRAM_ACC_ENERGY_PER_BIT = 0.1e-12
# DPU = dot product unit
NUM_DPU = 16
DPU_SIZE = 128
SINGLE_DPU_ENERGY_PER_CYCLE = 20e-12
CLOCK_PERIOD = 1e-9

DRAM_ACC_LATENCY = 7000
WEIGHT_SRAM_LOAD_LATENCY = 6
ACTIVATION_SRAM_LOAD_LATENCY = 3
ACTIVATION_SRAM_WRITE_LATENCY = 3
COMPUTE_LATENCY = 2
