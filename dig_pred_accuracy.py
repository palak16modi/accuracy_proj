#for finding accuracy of all the digit predictions

def pred_dig(a):
    a=int(a)
    y_train_a=(y_train==a)
    y_test_a=(y_test==a)
    sgd_a = SGDClassifier(random_state=42)
    sgd_a.fit(x_train,y_train_a)
    pred_test=sgd_a.predict(x_test)
    acc=sum(pred_test==y_test_a)/len(y_test)
    print("accuracy of prediction is: ", acc)
    
import sklearn
from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784",version=1)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
x=mnist['data']
y=mnist['target']
plt.imshow(x[0].reshape(28,28))
y=np.uint8(y)
x_train=x[0:60000]
x_test=x[60000:]
y_train=y[0:60000]
y_test=y[60000:]
a=int(input("enter digit "))
print("input taken, calculating.....")
    

pred_dig(a)

