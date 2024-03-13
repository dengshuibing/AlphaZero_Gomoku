from multiprocessing import Process
li = []

def foo(i):
    li.append(i)
    print('say hi', li)
if __name__ == '__main__':

    for i in range(10):
        p = Process(target=foo, args=(i,))
        p.start()

    print('ending', li)
