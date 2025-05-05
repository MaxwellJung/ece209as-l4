import part_a, part_b, part_c, part_d

def main():
    # Part a
    df_name = 'IS'
    batch_size = 1
    total_energy = part_a.input_stationary_total_energy(batch_size=batch_size)
    print(f"dataflow = {df_name}, batch size = {batch_size}, energy = {total_energy} J")

    df_name = 'IS'
    batch_size = 256
    total_energy = part_a.input_stationary_total_energy(batch_size=batch_size)
    print(f"dataflow = {df_name}, batch size = {batch_size}, energy = {total_energy} J")

    df_name = 'OS'
    batch_size = 1
    total_energy = part_a.output_stationary_total_energy(batch_size=batch_size)
    print(f"dataflow = {df_name}, batch size = {batch_size}, energy = {total_energy} J")

    df_name = 'OS'
    batch_size = 256
    total_energy = part_a.output_stationary_total_energy(batch_size=batch_size)
    print(f"dataflow = {df_name}, batch size = {batch_size}, energy = {total_energy} J")

    # Part b
    batch_size = 0
    latency_seconds = 0
    fps = 0
    print(f"batch size = {batch_size}, latency = {latency_seconds} s, average FPS = {fps}")

    # Part c
    dpu_size = 0
    fps = 0
    print(f"dot product unit size = {dpu_size}, average FPS = {fps}")

    # Part d
    batch_size = 0
    latency_seconds = 0
    fps = 0
    print(f"batch size = {batch_size}, pipelined latency = {latency_seconds} s, pipelined average FPS = {fps}")


if __name__ == '__main__':
    main()
