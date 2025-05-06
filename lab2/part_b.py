import math
from system_specs import *

def main():
    test()

def test():
    batch_size = 1
    latency_seconds = weight_stationary_total_latency(batch_size=batch_size)
    fps = batch_size/latency_seconds
    print(f"batch size = {batch_size}, latency = {latency_seconds} s, average FPS = {fps}")

    batch_size = 256
    latency_seconds = weight_stationary_total_latency(batch_size=batch_size)
    fps = batch_size/latency_seconds
    print(f"batch size = {batch_size}, latency = {latency_seconds} s, average FPS = {fps}")


def weight_stationary_total_latency(batch_size=1, dpu_size=DPU_SIZE):
    dram_latency = DRAM_ACC_LATENCY

    # Calculate latency for 5 convolutional layers
    conv_layer_latencies = [0] * 5
    for i in range(5):
        conv_layer_latencies[i] = calc_weight_stationary_conv_latency(**conv_layer_params[i], batch_size=batch_size, dpu_size=dpu_size)

    total_latency = sum(conv_layer_latencies) + dram_latency

    return total_latency


def calc_weight_stationary_conv_latency(
        input_width=32, input_height=32,
        num_filter=32, num_channel=3, filter_width=3, filter_height=3, 
        batch_size=1, dpu_size=DPU_SIZE):

    # Convert convolution to matrix-matrix multiply, i.e. WI = O where
    # shape(W) = num_filter x (num_channel x filter_width x filter_height) = 32 x (3 x 3 x 3) = 32 x 27
    # shape(I) = (num_channel x filter_width x filter_height) x (input_width x input_height) = (3 x 3 x 3) x (32 x 32) = 27 x 1024
    # shape(O) = 32 x 1024
    # See lecture 2 slide 6 for illustration of im2col
    return calc_weight_stationary_matrix_mult_latency(weight_rows=num_filter, input_rows=num_channel*filter_width*filter_height, input_cols=input_width*input_height*batch_size, dpu_size=dpu_size)


def calc_weight_stationary_matrix_mult_latency(weight_rows=32, input_rows=27, input_cols=1024, dpu_size=DPU_SIZE):
    # Timing diagram if activation SRAM is dual ported
    # (4 input updates per weight update)
    # Weight   ------                 ------                    
    # Input    ---     ---  ---  ---  ---     ---  ---  ---     
    # Compute        --   --   --   --      --   --   --   --   
    # Output           ---  ---  ---  ---     ---  ---  ---  ---

    # Timing diagram if activation SRAM is single ported
    # (4 input updates per weight update)
    # Weight   ------                             ------                             
    # Input    ---        ---     ---     ---     ---        ---     ---     ---     
    # Compute        --      --      --      --         --      --      --      --   
    # Output           ---     ---     ---     ---        ---     ---     ---     ---

    weight_buf_update_count = math.ceil(weight_rows/NUM_DPU) * math.ceil(input_rows/dpu_size)
    input_buf_update_count_per_weight_update = input_cols

    latency_per_weight_update = WEIGHT_SRAM_LOAD_LATENCY + \
        input_buf_update_count_per_weight_update*COMPUTE_LATENCY + \
        input_buf_update_count_per_weight_update*ACTIVATION_SRAM_WRITE_LATENCY + \
        (input_buf_update_count_per_weight_update-1)*ACTIVATION_SRAM_LOAD_LATENCY
    
    total_latency = latency_per_weight_update * weight_buf_update_count

    return total_latency


if __name__ == '__main__':
    main()
