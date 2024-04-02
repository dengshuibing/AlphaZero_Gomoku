import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# 异步操作，变量+1，入队，出队列
Q = tf.FIFOQueue(100, dtypes=tf.float32)

# 要做的事情
var = tf.Variable(0.0)
data = tf.assign_add(var, 1)
en_q = Q.enqueue(data)

# 队列管理器op
qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 5)

# 变量初始化op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化变量
    sess.run(init_op)

    # 开启线程协调器
    coord = tf.train.Coordinator()

    # 开始子线程
    threads = qr.create_threads(sess, coord=coord, start=True)

    # 主线程读取数据
    for i in range(50):
        print(sess.run(Q.dequeue()))

    # 请求停止线程
    coord.request_stop()

    coord.join()
