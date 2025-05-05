import math
from system_specs import *

def main():
    # Calculate WS energy for 6th layer, i.e. Conv [512, 256, 3, 3]
    ws_total_energy = ws_conv_energy(
        input_width=16, input_height=16,
        num_filter=512, num_channel=256, filter_width=3, filter_height=3,
        batch_size=1)

    print(f'{input_stationary_total_energy(batch_size=1)=}')
    print(f'{input_stationary_total_energy(batch_size=256)=}')

    print(f'{output_stationary_total_energy(batch_size=1)=}')
    print(f'{output_stationary_total_energy(batch_size=256)=}')


def input_stationary_total_energy(batch_size=1):
    # Count total number of weight params in network
    weight_param_count = 0
    for i in range(5):
        weight_param_count += conv_layer_params[i]['filter_width']*conv_layer_params[i]['filter_height']*conv_layer_params[i]['num_channel']*conv_layer_params[i]['num_filter']
    
    # Energy to move weight params from DRAM to SRAM
    weight_dram_energy = DRAM_ACC_ENERGY_PER_BIT * (weight_param_count * BITS_PER_WEIGHT)
    input_param_count = 32 * 32 * 3
    # Energy to move input params from DRAM to SRAM
    input_dram_energy = DRAM_ACC_ENERGY_PER_BIT * (input_param_count * batch_size * BITS_PER_INPUT)
    dram_energy = input_dram_energy + weight_dram_energy

    # Calculate energy for 5 convolutional layers
    conv_layer_energies = [0] * 5
    for i in range(5):
        conv_layer_energies[i] = calc_input_stationary_conv_energy(**conv_layer_params[i], batch_size=batch_size)

    total_energy = sum(conv_layer_energies) + dram_energy

    return total_energy


def calc_input_stationary_conv_energy(
        input_width=32, input_height=32,
        num_filter=32, num_channel=3, filter_width=3, filter_height=3, 
        batch_size=1):
    
    # Convert convolution to matrix-matrix multiply, i.e. WI = O where
    # shape(W) = num_filter x (num_channel x filter_width x filter_height) = 32 x (3 x 3 x 3) = 32 x 27
    # shape(I) = (num_channel x filter_width x filter_height) x (input_width x input_height) = (3 x 3 x 3) x (32 x 32) = 27 x 1024
    # shape(O) = 32 x 1024
    # See lecture 2 slide 6 for illustration of im2col
    return calc_input_stationary_matrix_mult_energy(weight_rows=num_filter, input_rows=num_channel*filter_width*filter_height, input_cols=input_width*input_height*batch_size)


def calc_input_stationary_matrix_mult_energy(weight_rows=32, input_rows=27, input_cols=1024):

    # Matrix multiply WI = (32 x 27) (27 x 1024) = (32 x 1024)
    # Note that compute module is limited to Av = (16 x 128) (128 x 1) every cycle

    input_buf_update_count = math.ceil(input_rows/DPU_SIZE) * input_cols
    weight_buf_update_count_per_input_update = math.ceil(weight_rows/NUM_DPU)

    dot_prod_cycle_count = weight_buf_update_count_per_input_update * input_buf_update_count
    dot_product_energy = SINGLE_DPU_ENERGY_PER_CYCLE * dot_prod_cycle_count * NUM_DPU
    
    weight_buf_update_count = weight_buf_update_count_per_input_update * input_buf_update_count
    # number of weights in weight buffer
    weight_buf_size = NUM_DPU * DPU_SIZE
    weight_sram_energy = SRAM_ACC_ENERGY_PER_BIT * (BITS_PER_WEIGHT * weight_buf_size) * weight_buf_update_count

    input_buf_size = DPU_SIZE
    input_sram_energy = SRAM_ACC_ENERGY_PER_BIT * (BITS_PER_INPUT * input_buf_size) * input_buf_update_count

    output_buf_size = NUM_DPU
    # Every dot product creates new partial output
    output_buf_update_count = dot_prod_cycle_count
    output_sram_write_energy = SRAM_ACC_ENERGY_PER_BIT * (BITS_PER_OUTPUT * output_buf_size) * output_buf_update_count
    output_sram_read_energy = ((math.ceil(input_rows/DPU_SIZE)-1)/math.ceil(input_rows/DPU_SIZE)) * output_sram_write_energy
    output_sram_energy = output_sram_write_energy + output_sram_read_energy

    total_energy = weight_sram_energy + input_sram_energy + output_sram_energy + dot_product_energy

    return total_energy


def output_stationary_total_energy(batch_size=1):
    # Count total number of weight params in network
    weight_param_count = 0
    for i in range(5):
        weight_param_count += conv_layer_params[i]['filter_width']*conv_layer_params[i]['filter_height']*conv_layer_params[i]['num_channel']*conv_layer_params[i]['num_filter']
    
    # Energy to move weight params from DRAM to SRAM
    weight_dram_energy = DRAM_ACC_ENERGY_PER_BIT * (weight_param_count * BITS_PER_WEIGHT)
    input_param_count = 32 * 32 * 3
    # Energy to move input params from DRAM to SRAM
    input_dram_energy = DRAM_ACC_ENERGY_PER_BIT * (input_param_count * batch_size * BITS_PER_INPUT)
    dram_energy = input_dram_energy + weight_dram_energy

    # Calculate energy for 5 convolutional layers
    conv_layer_energies = [0] * 5
    for i in range(5):
        conv_layer_energies[i] = calc_output_stationary_conv_energy(**conv_layer_params[i], batch_size=batch_size)

    total_energy = sum(conv_layer_energies) + dram_energy

    return total_energy


def calc_output_stationary_conv_energy(
        input_width=32, input_height=32,
        num_filter=32, num_channel=3, filter_width=3, filter_height=3, 
        batch_size=1):
    
    # Convert convolution to matrix-matrix multiply, i.e. WI = O where
    # shape(W) = num_filter x (num_channel x filter_width x filter_height) = 32 x (3 x 3 x 3) = 32 x 27
    # shape(I) = (num_channel x filter_width x filter_height) x (input_width x input_height) = (3 x 3 x 3) x (32 x 32) = 27 x 1024
    # shape(O) = 32 x 1024
    # See lecture 2 slide 6 for illustration of im2col
    return calc_output_stationary_matrix_mult_energy(weight_rows=num_filter, input_rows=num_channel*filter_width*filter_height, input_cols=input_width*input_height*batch_size)


def calc_output_stationary_matrix_mult_energy(weight_rows=32, input_rows=27, input_cols=1024):

    # Matrix multiply WI = (32 x 27) (27 x 1024) = (32 x 1024)
    # Note that compute module is limited to Av = (16 x 128) (128 x 1) every cycle

    filters_per_dpu = math.ceil(weight_rows/NUM_DPU)
    output_vect_count = filters_per_dpu*input_cols

    partial_sum_count_per_output_vect = math.ceil(input_rows/DPU_SIZE)
    dot_prod_cycle_count = partial_sum_count_per_output_vect*output_vect_count
    dot_product_energy = SINGLE_DPU_ENERGY_PER_CYCLE * dot_prod_cycle_count * NUM_DPU
    
    weight_buf_update_count = partial_sum_count_per_output_vect*output_vect_count
    # number of weights in weight buffer
    weight_buf_size = NUM_DPU * DPU_SIZE
    weight_sram_energy = SRAM_ACC_ENERGY_PER_BIT * (BITS_PER_WEIGHT * weight_buf_size) * weight_buf_update_count

    input_buf_update_count = partial_sum_count_per_output_vect*output_vect_count
    input_buf_size = DPU_SIZE
    input_sram_energy = SRAM_ACC_ENERGY_PER_BIT * (BITS_PER_INPUT * input_buf_size) * input_buf_update_count

    output_buf_size = NUM_DPU
    # No need to write partial sum to SRAM
    # Only 1 write per output vector
    output_buf_update_count = output_vect_count
    output_sram_write_energy = SRAM_ACC_ENERGY_PER_BIT * (BITS_PER_OUTPUT * output_buf_size) * output_buf_update_count
    output_sram_read_energy = 0
    output_sram_energy = output_sram_write_energy + output_sram_read_energy

    total_energy = weight_sram_energy + input_sram_energy + output_sram_energy + dot_product_energy

    return total_energy


def ws_conv_energy(input_width=32, input_height=32,
                   num_filter=32, num_channel=3, filter_width=3, filter_height=3, 
                   batch_size=1):

    kernel_size = filter_width * filter_height
    output_size_per_filter = input_width * input_height

    filters_per_dpu = math.ceil(num_filter / NUM_DPU)
    dpu_required_per_output = math.ceil(num_channel * kernel_size / DPU_SIZE)
    num_weight_buf_updates = filters_per_dpu * dpu_required_per_output
    num_dot_prod_cycles = num_weight_buf_updates * output_size_per_filter

    weight_sram_energy = num_weight_buf_updates * NUM_DPU * DPU_SIZE * WEIGHT_BITWIDTH * SRAM_ACC_ENERGY_PER_BIT

    input_sram_energy = num_dot_prod_cycles * DPU_SIZE * ACTIVATION_BITWIDTH * SRAM_ACC_ENERGY_PER_BIT

    output_sram_write_energy = num_dot_prod_cycles * NUM_DPU * ACTIVATION_BITWIDTH * SRAM_ACC_ENERGY_PER_BIT
    output_sram_read_energy = ((dpu_required_per_output-1)/dpu_required_per_output) * output_sram_write_energy
    output_sram_energy = output_sram_write_energy + output_sram_read_energy

    dot_product_energy = num_dot_prod_cycles * NUM_DPU * SINGLE_DPU_ENERGY_PER_CYCLE

    total_energy = weight_sram_energy + input_sram_energy + output_sram_energy + dot_product_energy

    return total_energy


if __name__ == '__main__':
    main()
