import math
from system_specs import *
import part_b
import matplotlib.pyplot as plt

def main():
    test()


def test():
    dpu_sizes = [128, 256, 512, 1024, 2048]
    fps_list = []

    for dpu_size in dpu_sizes:
        batch_size = 1
        latency_seconds = part_b.weight_stationary_total_latency(batch_size=batch_size, dpu_size=dpu_size)
        fps = batch_size/latency_seconds
        fps_list.append(fps)
        print(f"dot product unit size = {dpu_size}, average FPS = {fps}")

    plt.plot(dpu_sizes, fps_list)
    plt.xlabel("Dot Product Unit Size")
    plt.ylabel("average FPS")
    plt.show()


if __name__ == '__main__':
    main()
