import time

MEGABYTE = 1048576


# DoS emulation through memory exhaustion
# target = target number of bytes
# increment = bytes to increment for every iteration
# sleep = seconds to sleep between iterations
def memory_exhaustion(target, increment):
    # s = variable using memory
    bytes = 0  # current number of bytes used
    sleepPeriod = 0.1

    # Convert input megabytes to bytes
    target = target * MEGABYTE
    increment = increment * MEGABYTE

    # Calculate increment for each interation
    increment = increment * sleepPeriod
    increment = int(increment)

    while bytes < target:
        s = "A" * bytes
        # print bytes
        bytes = bytes + increment
        time.sleep(sleepPeriod)

    exit(0)


if __name__ == "__main__":
    memory_exhaustion(4000, 100)
