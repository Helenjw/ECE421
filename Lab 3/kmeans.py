# Distance function for K-means
def distanceFunc(X, MU):
    X = tf.expand_dims(X, 1)
    MU = tf.expand_dims(MU, 0)
    pair_distance = tf.reduce_sum(tf.square(tf.subtract(X, MU)), 2)
    return pair_distance
 
class KMeans(object):
    def __init__(self, K, D):
        self.X = tf.placeholder(shape=[None, D], dtype=tf.float64)
        self.mu = tf.Variable(tf.random_normal(shape=[K,D],dtype=tf.float64), trainable=True, dtype=tf.float64)
        pair_distance = distanceFunc(self.X, self.mu)
        self.classes = tf.argmin(pair_distance, 1)
        self.loss = tf.reduce_sum(tf.reduce_min(pair_distance,1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    def train(self, data):
        feed_dict = {self.X : data}
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict)
        return loss
    def evaluate(self, data):
        feed_dict = {self.X : data}
        feed_dict = {self.X : data}
        loss, classes = self.sess.run([self.loss, self.classes], feed_dict)
        return loss, classes
    def get_final_params(self, K):
        print("Number of Clusters: {}".format(K))
        mean = self.sess.run([self.mu])
        print("mu's: {}".format(mean[0]))
        return mean
 
K = 3
model = KMeans(K, dim)
loss = []
 
for i in range(1000):
    loss.append(model.train(data))
mean = model.get_final_params(K)
plt.plot(np.arange(len(loss)), loss, 'k')
plt.xlabel("Number of updates")
plt.ylabel("Loss")
plt.title("Loss with K={}".format(K))
plt.show()
plt.clf()
train_loss, train_classes = model.evaluate(data)
plt.scatter(data[:, 0], data[:, 1], c=train_classes, s=50, alpha=0.5)
plt.plot(mean[0][:, 0], mean[0][:, 1], 'kx', markersize=10)
plt.title("Training Data with K={}".format(K))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.clf()


for K in [1,2,3,4,5]:
    model = KMeans(K, dim)
    loss = []
    for i in range(4000):
        loss.append(model.train(data))
    mean = model.get_final_params(K)
    train_loss, train_classes = model.evaluate(data)
    percents = [0 for k in range(K)]
    for i in range(K):
        for c in train_classes:
            if c == i:
                percents[i] += 1
    for idx, ele in enumerate(percents):
        print("Percent of data points belonging to cluster {}: {}".format(idx+1,ele/len(train_classes)))
    plt.scatter(data[:, 0], data[:, 1], c=train_classes, s=50, alpha=0.5)
    plt.plot(mean[0][:, 0], mean[0][:, 1], 'kx', markersize=10)
    plt.title("Training Data with K={}".format(K))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
