#!/usr/bin/env python
import time

if __name__ == '__main__':
    a = 0
    start = time.time()
    for i in range(100000):
        for j in range(100000):
            a = 1
    end = time.time()
    print(end - start)
    print("hello")
