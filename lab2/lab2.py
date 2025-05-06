import part_a, part_b, part_c, part_d

def main():
    # Part a
    part_a.test()

    # Part b
    part_b.test()

    # Part c
    part_c.test()

    # Part d
    batch_size = 0
    latency_seconds = 0
    fps = 0
    print(f"batch size = {batch_size}, pipelined latency = {latency_seconds} s, pipelined average FPS = {fps}")


if __name__ == '__main__':
    main()
