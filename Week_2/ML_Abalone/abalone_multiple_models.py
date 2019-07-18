from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

p = Perceptron()
dt = DecisionTreeClassifier(random_state=0)
neigh = KNeighborsClassifier(weights='distance')
ovr = OneVsRestClassifier(LinearSVC(random_state=0))

trainingdata = open("training_data.txt","r")


x = [] #features (values)
y = [] #label



for line in trainingdata:
    y.append(line[-3:-1].replace(",","")) #egyjegyu a szam eseten a masodik karakter ','
    x.append(line[0:len(line)-4].split(","))


trainingdata.close()

# M = 0, F = 1, I = 2 (későbbi konvertálás miatt)

for i in range(0,len(x)):
    if x[i][0] == 'M':
        x[i][0] = 0
        
    elif x[i][0] == 'F':
        x[i][0] = 1

    else:
        x[i][0] = 2


# abalone.names emlitette, hogy ha 3 osztalyra bontjuk a 29-et akkor milyen eredmenyeket kapunk

index = 0
for num in y:
    num = int(num)
    
    if num > 0 and num < 9:
        y[index] = 0
        
    elif num == 9 or num == 10:
        y[index] = 1

    else:
        y[index] = 2
        
    index += 1


x = np.array(x,dtype = float)

#Ez lenne a tanulas

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


#========================================================Testing==============================================================



trainingdata = open("test_data.txt","r")


x = [] #features
y = [] #label



for line in trainingdata:
    y.append(line[-3:-1].replace(",",""))
    x.append(line[0:len(line)-4].split(","))

trainingdata.close()

for i in range(0,len(x)):
    if x[i][0] == 'M':
        x[i][0] = 0
        
    elif x[i][0] == 'F':
        x[i][0] = 1

    else:
        x[i][0] = 2


index = 0
for num in y:
    num = int(num)
    
    if num > 0 and num < 9:
        y[index] = 0
        
    elif num == 9 or num == 10:
        y[index] = 1
        
    else:
        y[index] = 2
        
    index += 1


x = np.array(x,dtype = float)

print("\n======================")
print("Testing Results")
print("======================\n")

print("Perceptron: ",accuracy_score(y,p.predict(x)))
print("OVR: ",accuracy_score(y,ovr.predict(x)))
print('DT: ', accuracy_score(y,dt.predict(x)))
print("KNN: ",accuracy_score(y,neigh.predict(x)))
