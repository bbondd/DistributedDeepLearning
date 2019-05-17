import ray
import time

@ray.remote
def f():
    time.sleep(10)

ray.get([f.remote() for _ in range(3)])