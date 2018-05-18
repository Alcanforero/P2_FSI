import gzip
import pickle as cPickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y.astype(int), 10)
test_x, test_y = test_set
test_y = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 28*28])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(28*28, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# ---------------- Visualizing some element of the MNIST dataset --------------
batch_size = 20
epoch = 0
errorActual = 1
errorAnterior = 100000
check = 1.0
validErrors = []
trainErrors = []

print(" ----------------------------------------------- ")
print("|\tEpoch\t|\tTrainError\t|  ValidationError\t|")
print(" ----------------------------------------------- ")
while (check > 0.01):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    trn = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    trainErrors.append(trn)

    errorActual = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    check = errorAnterior - errorActual
    validErrors.append(errorActual)
    errorAnterior = errorActual
    print("|\t", epoch, " \t|\t", trn, " \t|\t  ", errorActual, "  \t|")
    epoch += 1
    result = sess.run(y, feed_dict={x: valid_x})

print("-------------------------------------------------")

error = 0
results = sess.run(y, feed_dict={x: test_x})
for filasy, filasr in zip(test_y, results):
    if np.argmax(filasy) != np.argmax(filasr):
        error += 1
print("Errores en Test: ", error)

plt.suptitle("MNIST Convolutional Neural Network")
plt.title("Evolución del error en el entrenamiento")
x = np.arange(epoch)
plt.plot(x, validErrors, "b-", label="Errores según Validacion")
plt.plot(x, trainErrors, "r-", label="Errores según Entrenamiento")
plt.legend(loc=0)
plt.show()

plt.suptitle("MNIST Convolutional Neural Network")
plt.title("Evolución del error en el entrenamiento")
plt.plot(x, trainErrors, "r-", label="Errores según Entrenamiento")
plt.legend(loc=0)
plt.show()


