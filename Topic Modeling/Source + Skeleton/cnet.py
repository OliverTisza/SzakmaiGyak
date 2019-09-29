from time import time
import pickle as pkl
import argparse

"""

Kikeresi a cnetből az egyes glove-os szavak összes conceptjét.
Korlátozottságok miatt, csak az első 50k szó concepjét keresi ki. 57.-60. sor 

-------------

Paraméterek :

	path : ahonnan az i2w-t betölti, és ugyanide menti az új fájlt concepts.pkl néven
	cnet-location : ahol a cnet van
	
		én ezt még akkoriban kicsomagoltam, úgyhogy
		egy assertions.csv-t keres (conceptnet-assertions-5.7.0.csv.gz eredetileg)

-------------

Return :

	dict() : key = int
		 value = list of concepts

"""





parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', required=False,default=".", type=str)
parser.add_argument('--cnet-location', required=False,default=".", type=str)
args = parser.parse_args()
path = args.path
cnet_location = args.cnet_location

f = open(cnet_location+'\\assertions.csv','r',encoding="utf8")

with open(path+'\\unprocessed_ws.pkl','rb') as g:
	i2w = pkl.load(g)

t = time()

word_concept = dict()

i = 0
"""
-------------
init
-------------
"""
for k in i2w:
	word_concept.update({ i2w[k] : []})
	#print(i)
	i += 1
	if i == 50000:
		break
"""
-------------
kikeres
-------------
"""


line = f.readline().split()
while line:
	if line[2][6:] in word_concept.keys() and ('/c/en/' in line[3] and '/c/en/' in line[2]):
		#print(line[2][6:],line[3][6:])
		word_concept[line[2][6:]].append(line[3][6:])

	elif line[3][6:] in word_concept.keys() and ('/c/en/' in line[3] and '/c/en/' in line[2]):
		#print(line[3][6:],line[2][6:])
		word_concept[line[3][6:]].append(line[2][6:])

							
	line = f.readline().split()

pkl.dump(word_concept,open(path+"\\concepts.pkl","wb"))
print('Cnet time: {} mins'.format(round((time() - t) / 60, 2)))
