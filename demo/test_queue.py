import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# 同步操作，如队列，+1，出队列

# 创建一个队列
Q = tf.queue.FIFOQueue(3, dtypes=tf.float32)

# 数据进队列
init_q = Q.enqueue_many([[1.0, 2.0, 3.0], ])

# 定义操作
de_q = Q.dequeue()
data = de_q + 1
en_q = Q.enqueue(data)

with tf.Session() as sess:

    # 初始化队列
    sess.run(init_q)

    # 执行10次 +1 操作
    for i in range(10):
        sess.run(en_q)

    # 取出数据
    for i in range(Q.size().eval()):
        print(Q.dequeue().eval())

