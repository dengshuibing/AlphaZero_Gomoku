import multiprocessing
import time
 
 
def func(msg):
    print("in:", msg)
    time.sleep(3)
    print("out,", msg)
    return msg
 
if __name__ == "__main__":
    # 这里设置允许同时运行的的进程数量要考虑机器cpu的数量，进程的数量最好别小于cpu的数量，
    # 因为即使大于cpu的数量，增加了任务调度的时间，效率反而不能有效提高
    pool = multiprocessing.Pool()
    item_list = ['processes1' ,'processes2' ,'processes3' ,'processes4' ,'processes5' ,]
    count = len(item_list)
    results = []
    a = time.time()
    for item in item_list:
        msg = "python教程 %s" %item
        # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
        res = pool.apply_async(func, (msg,))
        # print("res==="+res.get())
        results.append(res)
    
    pool.close()
    pool.join() # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print("cost time: "+str(time.time() - a)+" s")
    for result in results:
        print(result.get())