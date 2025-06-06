# Input shape is 3x32x32
# Network Definition
# Format: Conv [#F, #C, X, Y] -> number of filters, number of channels, filter width, filter height
# 1. Conv [128,3,3,3]
# 2. Conv [128,128,3,3]
# 3. Conv [128,128,3,3]
# 4. Conv [128,128,3,3]

NETWORK_LAYERS_1 = [
    {'input_width':32, 'input_height':32,
     'num_filter':128, 'num_channel':3, 'filter_width':3, 'filter_height':3},
    {'input_width':30, 'input_height':30,
     'num_filter':128, 'num_channel':128, 'filter_width':3, 'filter_height':3},
    {'input_width':28, 'input_height':28,
     'num_filter':128, 'num_channel':128, 'filter_width':3, 'filter_height':3},
    {'input_width':26, 'input_height':26,
     'num_filter':128, 'num_channel':128, 'filter_width':3, 'filter_height':3, 'final_layer': True},
]

NETWORK_LAYERS_2 = [
    {'input_width':32, 'input_height':32,
     'num_filter':128, 'num_channel':3, 'filter_width':3, 'filter_height':3},
    {'input_width':30, 'input_height':30,
     'num_filter':128, 'num_channel':128, 'filter_width':3, 'filter_height':3},
    {'input_width':28, 'input_height':28,
     'num_filter':128, 'num_channel':128, 'filter_width':3, 'filter_height':3},
    {'input_width':26, 'input_height':26,
     'num_filter':128, 'num_channel':128, 'filter_width':3, 'filter_height':3, 'final_layer': True},
]

# System Specs
BITS_PER_WEIGHT = 16
OPS_PER_MAC = 2
OPS_PER_SEC = 50e9 # 50 giga operations per second
BITS_PER_SEC = 2e9
