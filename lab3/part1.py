from system_specs import *

def main():
    tp_1_latency, tp_1_traffic = part1_q1_tensor_parallelism(NETWORK_LAYERS_1, input_count=1)
    pp_1_latency, pp_1_traffic = part1_q1_pipeline_parallelism(NETWORK_LAYERS_1, input_count=1)

    tp_32_latency, tp_32_traffic = part1_q1_tensor_parallelism(NETWORK_LAYERS_1, input_count=32)
    pp_32_latency, pp_32_traffic = part1_q1_pipeline_parallelism(NETWORK_LAYERS_1, input_count=32)

    q3_tp_32_latency, q3_tp_32_traffic = part1_q1_tensor_parallelism(NETWORK_LAYERS_2, input_count=32)
    q3_pp_32_latency, q3_pp_32_traffic = part1_q1_pipeline_parallelism(NETWORK_LAYERS_2, input_count=32)

    print(f"Q1: TP total latency: {tp_1_latency} ms")
    print(f"Q1: TP total network traffic: {tp_1_traffic} KB")
    print(f"Q1: PP total latency: {pp_1_latency} ms")
    print(f"Q1: PP total network traffic: {pp_1_traffic} KB")
    print(f"Q2: TP total latency: {tp_32_latency} ms")
    print(f"Q2: TP total network traffic: {tp_32_traffic} KB")
    print(f"Q2: PP total latency: {pp_32_latency} ms")
    print(f"Q2: PP total network traffic: {pp_32_traffic} KB")
    print(f"Q3: TP total latency: {q3_tp_32_latency} ms")
    print(f"Q3: TP total network traffic: {q3_tp_32_traffic} KB")
    print(f"Q3: PP total latency: {q3_pp_32_latency} ms")
    print(f"Q3: PP total network traffic: {q3_pp_32_traffic} KB")


def part1_q1_tensor_parallelism(network_layers, input_count=32):
    layer_latency_list = []
    layer_traffic_list = []

    for layer in network_layers:
        latency, traffic = calc_tp_layer(**layer, nodes=4)

        layer_latency_list.extend(latency)
        layer_traffic_list.append(traffic)

    total_latency_seconds = input_count*sum(layer_latency_list)
    total_traffic_bits = input_count*sum(layer_traffic_list)
    latency_ms = total_latency_seconds * 1e3
    traffic_kbits = total_traffic_bits / 1e3

    return latency_ms, traffic_kbits


def part1_q1_pipeline_parallelism(network_layers, input_count=32):
    layer_latency_list = []
    layer_traffic_list = []

    for layer in network_layers:
        latency, traffic = calc_pp_layer(**layer, nodes=4)

        layer_latency_list.extend(latency)
        layer_traffic_list.append(traffic)
    layer_latency_list.pop()
    layer_traffic_list.pop()

    # print(f'{layer_latency_list=}')
    # print(f'{layer_traffic_list=}')

    pipeline_period = max(layer_latency_list)

    total_latency_seconds = pipeline_period*(input_count-1) + sum(layer_latency_list)
    total_traffic_bits = input_count*sum(layer_traffic_list)
    latency_ms = total_latency_seconds * 1e3
    traffic_kbits = total_traffic_bits / 1e3

    return latency_ms, traffic_kbits


def calc_tp_layer(
        num_filter=128, 
        num_channel=3, 
        filter_width=3, 
        filter_height=3, 
        input_width=32, 
        input_height=32, 
        final_layer=False,
        nodes=4):

    filters_per_node = num_filter/nodes
    output_width = input_width - filter_width + 1
    output_height = input_height - filter_height + 1
    output_count_per_node = output_width * output_height * filters_per_node

    mac_per_output = filter_width * filter_height * num_channel
    mac_per_node = mac_per_output * output_count_per_node
    ops_per_node = mac_per_node * OPS_PER_MAC
    compute_latency_per_node = ops_per_node / OPS_PER_SEC

    output_bits_per_node = output_count_per_node * BITS_PER_WEIGHT
    output_broadcast_latency_per_node = output_bits_per_node / BITS_PER_SEC
    latency = (compute_latency_per_node, output_broadcast_latency_per_node)

    if final_layer:
        # each node broadcasts its output to a single output node
        output_broadcast_bits_per_node = output_bits_per_node
        # output node doesn't need to broadcast
        traffic = output_broadcast_bits_per_node * (nodes-1)
    else:
        # each node broadcasts its output to all other nodes
        output_broadcast_bits_per_node = output_bits_per_node * (nodes-1)
        # repeat for each node
        traffic = output_broadcast_bits_per_node * nodes

    return latency, traffic


def calc_pp_layer(
        num_filter=128, 
        num_channel=3, 
        filter_width=3, 
        filter_height=3, 
        input_width=32, 
        input_height=32, 
        final_layer=False,
        nodes=4):

    filters_per_node = num_filter
    output_width = input_width - filter_width + 1
    output_height = input_height - filter_height + 1
    output_count_per_node = output_width * output_height * filters_per_node

    mac_per_output = filter_width * filter_height * num_channel
    mac_per_node = mac_per_output * output_count_per_node
    ops_per_node = mac_per_node * OPS_PER_MAC
    compute_latency_per_node = ops_per_node / OPS_PER_SEC

    output_bits_per_node = output_count_per_node * BITS_PER_WEIGHT
    if final_layer:
        output_broadcast_latency_per_node = 0
    else:
        output_broadcast_latency_per_node = output_bits_per_node / BITS_PER_SEC
    latency = (compute_latency_per_node, output_broadcast_latency_per_node)

    if final_layer:
        # output node doesn't need to broadcast
        output_broadcast_bits = 0
        traffic = output_broadcast_bits
    else:
        # each node broadcasts its output to next node
        output_broadcast_bits = output_bits_per_node * 1
        traffic = output_broadcast_bits

    return latency, traffic


if __name__ == '__main__':
    main()
