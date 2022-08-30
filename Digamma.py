import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
arr=tf.range(1.0,501.0)/10
mass=0.5772156649
@tf.function
def ln_gamma(x):
    g=0.0
    for k in range(1,1001):
        g+=((x/k)-tf.math.log(1+(x/k)))
    return -mass*x-tf.math.log(x)+g
def digamma(arr):
  x=tf.Variable(arr,dtype=tf.float32)
  with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(x)
      y=tf.constant(ln_gamma(x))
  dig=tape.gradient(y,x).numpy()
  del tape
  return dig
dig=digamma(arr)
print(dig)
plt.plot(arr,dig,color='r')
plt.show()
