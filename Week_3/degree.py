import matplotlib.pyplot as plt
import math

f = open("assertions.csv",'r',encoding="utf8")

mydict = {} #csucsok es azok eleinek szama


for i in range(0,100000):
    x = f.readline()
    x = x.split()
    x = x[0].split('[')
    x = x[1].replace(']','')
    x = x.split(',')


    if x[1] in mydict:
        mydict[x[1]] += 1
    elif x[2] in mydict:
        mydict[x[2]] += 1
    else:
        mydict[x[1]] = 1
 
f.close()

values = list(mydict.values())

newdict = {}

for i in range(0,len(values)):
    if str(values[i]) in newdict:
        newdict[str(values[i])] += 1
    else:
        newdict[str(values[i])] = 1
    
        
x_edged = list(newdict.keys())

for i in range(0, len(x_edged)):
    x_edged[i] = math.log(float(x_edged[i]))
    #x_edged[i] = int(x_edged[i])

no_of_nodes = list(newdict.values())

ossz = sum(no_of_nodes)

for i in range(0, len(no_of_nodes)):
    no_of_nodes[i] = no_of_nodes[i] / ossz



plt.plot(x_edged,no_of_nodes,'ro')
plt.ylabel('nodes(k/all)')
plt.xlabel('edges(log)')
plt.show()



