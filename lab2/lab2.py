import part_a

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


if __name__ == '__main__':
    main()
