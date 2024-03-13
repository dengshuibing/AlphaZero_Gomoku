from multiprocessing import Pool
import time

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(processes=4) as pool:         # start 4 worker processes
        result = pool.apply_async(f, (10,)) # evaluate "f(10)" asynchronously in a single process
        print(result.get(timeout=1))        # prints "100" unless your computer is *very* slow

        print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"

        it = pool.imap(f, range(10))
        print(next(it))                     # prints "0"
        print(next(it))                     # prints "1"
        print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow
        result = pool.apply_async(time.sleep, (10,))
        print(result.get(timeout=1))        # raises multiprocessing.TimeoutError
'''
100
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
0
1
4
Traceback (most recent call last):
  File "C:/Users/BruceWong/Desktop/develop/multiprocessingpool.py", line 19, in <module>
    print(next(res))
TypeError: 'MapResult' object is not an iterator

Process finished with exit code 1
'''
