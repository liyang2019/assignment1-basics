import multiprocessing as mp
from typing import Any
import time


def foo(i, j, ls, q):
    ls[i][0][0] = b'ab'
    q.put(f"{i} {j}")


if __name__ == "__main__":

    with mp.Manager() as manager:
        ls1 = manager.list([([b'a', b'a'], 1), ([b'a', b'b'], 2)])
        ls2 = manager.list([([b'a', b'a'], 1), ([b'a', b'b'], 2)])
        counter = manager.dict()
        q = mp.Queue()
        for j in range(3):
            p1 = mp.Process(target=foo, args=(0, j, ls1, q))
            p2 = mp.Process(target=foo, args=(1, j, ls2, q))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            print(ls1, type(ls1))
            print(ls2, type(ls2))
        while not q.empty():
            print(q.get())


    # ctx = mp.get_context("spawn")
    # q = ctx.Queue()
    # c = ctx.Condition()
    # processes = []
    # for i in range(2):
    #     p = ctx.Process(target=foo, args=(i, q, c))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # while not q.empty():
    #     print(q.get())

    # q = ctx.Queue()
    # processes = []
    # for j in range(2):
    #     p = ctx.Process(target=foo, args=(i, j, arr, q))
    #     for i in range(4):
    #         p = ctx.Process(target=foo, args=(i, j, arr, q))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()
    #     vals = []
    #     while not q.empty():
    #         vals.append(q.get())
    #         print(vals[-1])
    #     print(f"============{i}")
