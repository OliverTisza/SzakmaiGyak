from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
"""
f = open("balance-scale.data","r")

trainingdata = open("training_data.txt","w")
testdata = open("test_data.txt","w")

for i in range(0,499):
    trainingdata.write(f.readline())

for i in range(500,626):
    testdata.write(f.readline())


f.close()
trainingdata.close()
testdata.close()

print("job's done!")
"""
p = Perceptron(penalty='l2')
dt = DecisionTreeClassifier(random_state=0)
neigh = KNeighborsClassifier(weights='distance')
ovr = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=3000))

trainingdata = open("training_data.txt","r")

x = [] #features
y = [] #label



for line in trainingdata:
    y.append(line[0])
    x.append(line[2:9].split(","))

trainingdata.close()


x = np.array(x,dtype = int)

p.fit(x,y)
dt.fit(x,y)
neigh.fit(x, y)
ovr.fit(x,y)

print("\n======================")
print("Training Results")
print("======================\n")

print("Perceptron: ",accuracy_score(y,p.predict(x)))
print("OVR: ",accuracy_score(y,ovr.predict(x)))
print('DT: ', accuracy_score(y,dt.predict(x)))
print("KNN: ",accuracy_score(y,neigh.predict(x)))

#------------------------------------TEST DATA--------------------------

trainingdata = open("test_data.txt","r")

x = [] #features
y = [] #label


for line in trainingdata:
    y.append(line[0])
    x.append(line[2:9].split(","))

trainingdata.close()


x = np.array(x,dtype = int)


print("\n======================")
print("Testing Results")
print("======================\n")

print("Perceptron: ",accuracy_score(y,p.predict(x)))
print("OVR: ",accuracy_score(y,ovr.predict(x)))
print('DT: ', accuracy_score(y,dt.predict(x)))
print("KNN: ",accuracy_score(y,neigh.predict(x)))
