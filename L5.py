
"""

In The Name of GOD
 Dec 19 2025

@author: Ali Pilehvar Meibody


L5


"""


'''
------------ Python Basic ---------------


Human (en) <-----Interface(python) ---> Machine (0,1)

Python --> language --> vocab, grammar


1- Python built in functions --> print() input() type() len() open() abs()


2- Keywords --> Logice python ro taghir --> az bala b paeen , chab b rast man code ro run mikonam

SHoroote --> shoam mikhay y code ya chand khat code --> ejra nshe bvaraye , sharti depen

2.1. Just If --> rahzan fght yeseria. age shart True bood Anjam mide ag nabod bikhial
if shart:
    dastor 1 
    datsor2
2.2. IF else --> dorahie --> ag shart true bod yeseri run ag nabod yeseri
if shart:
    dastoor1
    dastoor2...
els:
    dastorate dg
    
2.3. if elif else --> dorahiaye too dar too

if shart1:
    dastor1
elif shart2:
    dastoor2
else:
    #na sharte 1 nas harte 2
    dastoor3
    
    
2.3. Loops ha shodim

2.3.1. For
for shoamrande in list :
    code
    codha ro

for i in [1,2,3,4,5]:
    code
    
i=1 --> code
i=2 -->code ....


for i in range(0,10 --)  --> range(start,end+1,step)

static repeat
dynamic repat
#iteration anjam bdi

mylist

for i in mylist:
    if .
    count
    append
    print()....
    
    
    
2.3.2. While --> ta zamani ke 
while --. for mibashad
i= start
while i <end:
    ....
    i = i + step
    



2.3.3. Functions --> Input ---> [box] ---> output

....






3-  Variable (moteghayer) zarf

3.1. Numbers (int,float, complex) --> ** * / + - , == != > < >= <=
3.2. Bool (True , False)
3.3.Str (string) --''  zarf[index] zarf[start:end+1] str.function()
       zarf.lower()   zarf.upper() emal nmishodan , khorojio midadan
       new_zarf = zarf.lower()
-  -  -   Iterables - - - - 
Chand value dakheel yekz arf (variable) brizim
3.4. List --> a = [10 ,1.3 , 'ali', True]  zarf[index] zarf[start:end+1]
      list.function() --> emal msiohodan khorji
      list.append('ali')    list.sort()  list.remove('dd')
    
      List [] --> ordered(index) changable , allow duplicated
3.5. Tuple ()->  ordred (index)  unchangable , allow duplicated
3.6. Set {} --> unordered(index) unchangable, No duplicated
3.7. Dict {} --> index value  || key value  a = {'esm' : 'ali' , 'nomre': 20 }  a['esm']






'''





def newton(m,a):
    f= m*a
    return f


zarf = newton(10,2)


'''


definition --> sakhtan

def esm(vorodi1,vorodi2,....):
    badande
    karahae k mikhay
    
    return yechizi



call --> sedash bzni use koni

zarf = esm(....)


'''


#nokteye jadid
def newton(m,a):
    f = m * a
    c = f + 10
    
    return f , c


out1 , out2 = newton(10,2)





#ma tavabe ro dar yek script minevisim (.py)

#importesh mikonim va use mikonimesh



'''

AMpro.py --> dota tabe VED() , Density()


----1----------
import Ampro


Ampro.VED() 



----2 ------

from Ampro import VED

VED()


---- 3-----

from AMpro import VED as V


V()



'''


#----fucntion --> class


class Nam:
    def __init__(self):
        pass
    
    
    def func1(self):
        pass
    
    def func2(self):
        pass
    
    def fucnt3(self):
        pass
    
    

#class --> too delehs koli tabe dare khob?
#yedone object besaz az classs


zarf = Nam()   #vorodi nam(a,b,c,c)


#az tavabeye dkahelsh estefade koni
zarf.func1()

zarf.func2()


#---3vome code zani
#-----
name = 'ali' #object az class STR

#clas --> def --> str fucntions 

name.lower()

name.upper()

'''

1--> az bal ta paeen miakridm
2---> fucntion  based , --> fucntion --> improtesh
3--> class --> class.fucntions() 

'''


from Ampro import VED_CLass


zarf = VED_CLass() #Object


zarf.func1()

zarf.func2()

zarf.func3()



'''
1 script .py khodemon nvshtim va mikhyam brim jolo 




beshinim modelszi melt pool anjam bdim

modelsazie yek vakoneshe shimiaee 


def vakonesh():
    
    element 
    elemnt 
    

def arnius
    
    
def vonhuf


def gi


def entropy

def entalphy


def reaction():
    

#----------

#---> ANN --> artificial neural network 
shabakeye asabie masnooe

maghze --> layer haye neuron skahte shdoe

to har neuron yekseri moadelat anjam mishe -->

[n1] ------>

[n2] ------> [n4] ---> W --> [neuron]

[n3] ----->


activation ( w1 * n1 + w2 * n2 + w3* n3 + B)
Relu , tanh , sigmoid , .....

100 tra neuron 



#-----Python --> Ketabkhane ha

oragnization , afrad , goroh haye danehsgahi , sherkat ha ,.....


--> Ketabkhoone --> koli folder dare --> to haer folde koli script .py
--> too har .py koli tabe o class
to dar to beham chasbide dar nahayta


ejaze mide ba seda zadane tabeye madar --> 400 ta tabe baham monazam
riazi , logic , code , rooye cpu, gpu run bshe


sklearn --> ketabkhone --> [folder tooshe]
#az ketabkhoen sklearn foldere neural network
esme ye tabast MLPRegressor

 
from sklearn.neural_network import MLPRegressor
model = MLPRegressor()

model.fit(x,y)

#yek model train  midim b sadegi

#dah ta model train koni va modlei k balatairn accuracy dre ro vardari

model.fit(x,y)
print(model.accuracy)

model2.fit

model3.fit



list_accuraciues=[]
for i in range(0,10):
    new_model = MLPregressor()
    new_model.fit()
    my_accuracy = new_model.accuracy
   list_accuraciues.append(my_accuracy)

list_accuraciues= [10,20,30,40, 88 , 76]

for acc in list-accuracies:
    if acc>best_ac:
        print(acc)
    

#class

#object = class()

#object.fit()  #tabe ha --> emali  #khoroji drn

#object.fit() --> train mdie tamam

#predicted = object.predict(x) 


#list.append()


#new_zarf = zarf.lower()



ketabkhone ????





ketabkhone --> 140000



ketabkhone bydefault khode python dre

math
time
random



'''

import math

'''
GPT 


'''
#math --> math.py --> too delesh koli chizie

#yeki az chiza --> constant va adade



print(math.pi)     # 3.141592653589793


def my_fucntion(a,b):
    
    c= a * b 
    
    d = math.pi * c
    
    return d
    



print(math.e)      # 2.718281828459045
print(math.tau)    # 6.283185307179586

#math.pi() #--. function ()

#def fucntion(vorodi1,vorodi2,....)
math.pi #adade -->variable
#---------
#tavabe darim


x = -7.5
print(math.fabs(x))    # 7.5

x = 7.5

print(math.ceil(x))    #8
print(math.floor(x))    #7



math.pow(2,3) # 8.0


def my_pow(numb1,numb2):
    
    new_numb = numb1 ** numb2
    return new_numb


zarf = my_pow(2,3) #8




#ketabkhoneye math 
#pow

#bydefault khdoe python in ketabkhonaro dare

#improtesh koni nasbesh nist downlaod koni

import math

zarf = math.pow(2,3) # 8.0
    
    
    
    

print(math.exp(2))        # e^2
print(math.log(10))       # Natural log (ln)
print(math.log10(100))    # Base-10 log
print(math.log2(8))       # Base-2 log
    



print(math.sqrt(16))      # 4.0
print(math.pow(2, 3))     # 8.0



math.sqrt(100) # 10.0


#tabe sqrt (harchi toosh bzar) -->radikalesho



angle_rad = math.pi / 4   # 45 degrees in radians

print(math.sin(angle_rad))  # 0.707...
print(math.cos(angle_rad))  # 0.707...
print(math.tan(angle_rad))  # 1.0



print(math.factorial(5))   # 120
print(math.comb(5, 2))     # 10 combinations
print(math.perm(5, 2))     # 20 permutations



#-------------------------------

import time

time.time() #NameError: name 'time' is not defined

local_time = time.localtime()
print(local_time)

'''
time.struct_time(tm_year=2025, tm_mon=12, tm_mday=19, tm_hour=16, tm_min=38, tm_sec=35, tm_wday=4, tm_yday=353, tm_isdst=0)

'''


#-------------------------
import random

#float between 0 , 1
print(random.random()) #0.44465045882794396 ,0.0656747268928537



#az felan adad ta felan adad bhm bde

print(random.uniform(1, 10))  #4.301197091312808 , 4.301197091312808



items = ['apple', 'banana', 'cherry']
print(random.choice(items))  #banana



items = ['apple', 'banana', 'cherry']
print(random.choice(items))  #banana




items = ['apple', 'banana', 'cherry']
print(random.choice(items))  #apple



#------- kh kh basic hastand 

'''
Khode python naneveshte nadare

organziation ha neveshtanesh


140 000 darim


1- Numpy --> Mohasebat estefade msihe, araye ha , matrix ha
--> scipy , sympy --> hale maodelat numerical modeling

2- Matplotlib --> rasm 
---> seaborn --> rasm haye pishrafte tar

3- Pandas --> ketabkhone hast baraye kar ba dade , excel, csv , cleaning data


4- Sklearn --> core machine leanring --> chize asli


5- Tensorflow --> Google --> deep learning
6- Pytorch --> facebook --> deep learning
#tensorflow, pytorch --> 1000000 ketabkhone



aya alan man masalan bnvism

import numpy miare??


numpy ->narm afaar -->download koni

chijorie?




--->

Mac , Windows --> Terminal , shell 
core asli computeret harf bzni


GUI , CLI 



ls -->list mikone hamechio
cd --> miri too jae k bekhay

(base) apm@APMs-MacBook-Pro ~ 

(base) apm@APMs-MacBook-Pro ~ % cd desktop 

 (base) apm@APMs-MacBook-Pro desktop % 

 (base) apm@APMs-MacBook-Pro desktop % ls
 
 cd .. --> maity aghab
 
 
 
 
kole laptabete mage ma kolei environe,ment ba anaconda nasakhtim???

Env --> 

environmenteto faal koni

yek environment skahtye bodm DL

conda activate esme_env

(base) apm@APMs-MacBook-Pro ~ % conda activate DL

(DL) apm@APMs-MacBook-Pro ~ % 





(DL) apm@APMs-MacBook-Pro ~ % conda deactivate
(base) apm@APMs-MacBook-Pro ~ %





mac --> terminal
window --> win + r --> wt --> run as adminstartion

ye safe siah baz mishe (shell)
ls -->list
cd -->mitoni jaee bri
linux brid bekhonid --> mv . cat vim rm rf ,...


conda activate ENVIRONMENT -->vared mishe

conda deactivate --> kharej mishe

(base) -->(environemnt)



#----------
ketabkhoanto mrii google search mizni Pypi in sait marjae download



pip install numpy
pip install pandas
pip install matplotlib
pip install matplotlib.pyplot





'''


#import PyGamLab

import numpy

import pandas

import matplotlib.pyplot


#ag error dad --> yani doros nasb nashode

#ag kh var rfri 

'''


-----------Numpy----------------


list -->


'''

b=[10,20,30,40]
a= list([10,20,30,40])

a[1] #20

a[1:3] # [20, 30]

a[2] =500

print(a) #[10, 20, 500, 40]

a.reverse()

print(a) #[40, 500, 20, 10]

a.sort()

print(a) #[10, 20, 40, 500]


2 * a # [10, 20, 40, 500, 10, 20, 40, 500]


for element in a:
    answer = 4 * element
    
    
'''


Paye  ye list --> Python neveshte shode ast --> motasefane nesbat C , C++ ina kh kh kh kond hesab mishe


Numpy --->Number + Python --> omd gof chi mishe man
ye chiz mes elist besazm

array (araye ha)

numpy.array() ---> LIst --> c++ neveshte shdoe bashe


har kari k mikoni har done done + -

numpy array approx. 60x faster than python list --> yek operation

4 ta laye --> har laye 40 neuron --> 160 neuron

160 operation matrixi rooye data --> [ 100 ta sample] --> 5 cross validation


500 x 160 = 60000  operation * 60x -


1 bodi



2---> list ha fght y listan

jadval mikhay

radif , sotoon

10 0.98
20 0.97 
30 0.96

2D 2 bodi

numpuy array --> 0D (adad) , 1D (list) , 2d (matrix, jadval) , 3d , 4d , ......



operation haye makhsos

zarb, sin , cos () , statisticall , ...mohasbe e k fk koni
roye ye list az adad anjam bdi ro man popohsesh midm




'''

#-----Numpy --> array bejaye list estefade konim

#list estefade 

#mohasebat hamechi 
#sklearn,. pandas , pytroch (Tensor) --> Numpy daran kar mikonan efficiency in memory

'''


----> Numpy


---> Yekbar ino bayad install krd


numpy pip --> google

--> pip install numpy

mac : terminal |  Windows Win + R , wt 
ye safe siahe

conda activate En
pip install numpy
conda deactivate

numpy nasb mishe tooye environemnti k hastid (na base)
spydered hamoon environement varesh bshid numpy mishnase



numpy --> -.-.- --> versioning


2.10.5

2 --> kh moheme hamechish ba 1,3 frghd re

2.10
2.9 --> feature ezafe shdo

2.10.4
2.10.5 -->bugg hichvaght shoam nvinid raf shode




numpy --> depend 5 ta ketabkhon dg

pip install numpy




numpy --> lmn 4.6 

numpy 2.7 
sklearn --> nmumpy 2.8 


project machine learning --> 


environemnt besazi


baraye yadgiri
esme ketabkhone --> google

https://numpy.org

donbale -_> [documentation]



--> getting staarted (tutorial) -->avalie --> kefayat mikone

--> API reference tka take tabe hasho barat ovorde

GPT --> kh komak konde y done tabe donbaelsh 
(vorodi ) klhorji -->code mide



https://numpy.org/doc/stable/user/absolute_beginners.html


https://apmaii.github.io/pygamlab/quickstart_tutorial.html#appendix-a-working-with-numpy




'''

#numpy. ---> np

#in yekgharardade developere


#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt



import numpy


#too delsh y class --> array()

#list


a= list([10,20,30,40])


b = numpy.array([10,20,30,40])


print(type(b)) #<class 'numpy.ndarray'>


#----0D sefr bodi ------
#ye adad array 

d= 1

c = numpy.array(1) 



print(d) #1
print(type(d)) #<class 'int'> --> pythonic

#d + 10 ---> 0.001


print(c) #1
print(type(c)) #<class 'numpy.ndarray'>

#c + 10 ---> 0.0000000000000001

#oto delet array zarf . etelaat bht mdie
#number of dimension
#attribute

c.ndim  #0 
c.size #1
c.shape #()


#---- 1D --------

a = list([10,20,30,40])

print(a) #[10, 20, 30, 40]

print(type(a)) #<class 'list'>

a[1] #20




a[1:4] #[20, 30, 40]


a[1]=1000
print(a) #[10, 1000, 30, 40]



b = numpy.array([10,20,30,40])
print(b) #[10 20 30 40]
print(type(b)) #<class 'numpy.ndarray'>


b[1] # 20

b[1:4] #array([20, 30, 40])


b[1]=10000
print(b) #[   10 10000    30    40]


#attributes

b.ndim #: 1
b.size #4
b.shape #(4,)


import numpy

b = numpy.array([10,20,30])



import numpy as np

b = np.array([10,20,30,40])




#--------- 2D -----------
''' 
      sotoon1  sotoon2   sootoon3 
radif1
radif2 
radif3
radif4 



      column1  column2   column3 
row1
row2 
row3
row4 


10 20 30 40
50 60 70 80

      column1  column2   column3  column4 
row1
row2 


(radif avale , bad sotoon) -->numpy ino mige


'''

#** yek bar kafie 
#improt haro balaye code mikonan

import numpy as np

#step 1
d = np.array( )


#step 2 -->whoel numpy
d = np.array([      ])

#stpe 3 --> chanta radif mikhay?

#
'''
10 20 30 40
50 60 70 80



2 ta numpy array hast?

np.array([10,20,30,40])
np.array(p[50,60,70,80])

'''


d = np.array([ [] , []      ])


#stepe akahr
d = np.array([ [10,20,30,40] , [50,60,70,80]      ])


#-->tyekrar 

'''

2 ta radif 3 ta soton
#7 ta radif 1 soton

'''
 
d.ndim #2
d.size #8
d.shape # (2, 4)
# 2 ta radif, 4 ta sotoon
#(tedad radif , tedad sotoon)


'''

10 20 30 40
50 60 70 80



70 mikham dastrsi peyda konm



list, nump array1 d --> zarf[index]

zarf[2]

'''

d[1]  #array([50, 60, 70, 80])

#kole radifo mide



'''
10 20 30 40
50 60 70 80

'''

d[0] # array([10, 20, 30, 40])
d[1] # array([50, 60, 70, 80])


#yedone element mikham


'''

kodom radif, kodom soton

radife 1 , sootone 2


zarf[radif,soton]

'''

#access
d[1,2] #70


#change
d[1,2] = 9000


print(d)

'''
[[  10   20   30   40]
 [  50   60 9000   80]]

'''


#-----> 3D --> numpy documentation
#3d ->chant jjadval rooham stack 
#zarf[kodom jadval , kodom radif, kodom soton]

#----------------------------
#assignemnts--------------------

#sade tarin halat
import numpy as np
a = np.array([10,20,30,40])



#list --> numpy aray

b = [10,20,30,40]

a = np.array(b)


#---fght manually dasti mishe inkararo krd

#Inaj k miad b komak


#arange --0> range()

my_array = np.arange(0,100)

#yek numpuy array
my_array.ndim # 1
my_array.size # 100

print(my_array[0:10]) #[0 1 2 3 4 5 6 7 8 9]



my_array = np.arange(0,1000 , 3) #c++ ejra mishe


#yek range ta y range midi migi tike parash kone
#az 1 ta 20 -->100 ta taghsim kon --> 
my_array2 = np.linspace(1, 20 , 100)





#-----tavabe zeros
#zaros --> zero misaze
#too delesh yechi migire bename shape
#zarf --> khoroji
#1 bdoiu 10 ta element
np.zeros(10) #array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



# 2 ta radif , 5 ta soton -- hamasho 0
#(radif , soton)
zarf = np.zeros( (2,5))



#----zeros -->
#shape midi

zarf2 = np.ones( (2,5))


#dota vorodi migiran , shape , ba ch addadi

zarf3 = np.full( (2,5) , 23)



zarf4 = np.eye(4)


#----- Random ha hastan
#bsorat random ham numpy array bsazi


#0 ta 1 --> koli adade random 

#yek numpy aray 10 tae msiaze

np.random.random(10)

'''
array([0.31576382, 0.25775163, 0.56740303, 0.01757415, 0.98548526, 0.6139723 , 0.80065814, 0.07528007, 0.2932216 , 0.86458351])

'''


np.random.random( (2,3))

'''
array([[0.92512391, 0.1184038 , 0.78388638],
       [0.46105048, 0.32275184, 0.24692825]])

'''


#ye adad ta ye adad int , float

#y array shape (2,5)  adade random beyen 1 ta 10 poresh
np.random.randint(1,10 ,(2,5))

'''
array([[3, 9, 9, 3, 6],
       [7, 2, 6, 3, 1]])

'''


#hamey adad az 0 ta 10 y shansi dashte bashan
np.random.uniform(1,10 , (2,5))

'''
array([[8.73605379, 9.90072265, 4.93020792, 4.69862613, 1.43644328],
       [5.60079937, 3.79597104, 7.78063897, 3.26019311, 4.2867375 ]])

'''


#nnormal -->  yk adadi hoel mehvare on ta yek fasele ee
#miu --> miangin ,, 

np.random.normal(5,1, (2,5))

'''
array([[3.82206572, 4.24298317, 4.49129514, 3.86289098, 6.61353874],
       [3.92067207, 3.90495246, 4.36545034, 4.99426382, 5.50496087]])

'''



a= np.array([10,20,30])
b= np.array([40,50,60])


c = a + b 

print(c) #[50 70 90]


#math --> begone neveshte shode ks homa btonid

np.add()
np.subtract()
np.multiply()
np.divide()
np.power()
np.abs()


#_---- statistically

a= np.array([10,20,30])


a.min() #10

a.max() #30

a.sum() #60

np.var()
np.prod()
np.std()
np.max()
np.min()
np.mean()
np.median()
np.percentile()

#--------

np.sin()

np.cos()

np.tan()


#------


a = np.arange(1,201)

#1 ta 200 ro misaze


a.ndim #1
a.shape # (200,)

#ino tabdil konm b ychizi k 2 ta radif, 100 ta sooton ?

#reshape()

b = a.reshape(2,100)


c = a.reshape(4,50)


#list ---> numpuy array ha neveshte mishe

'''

10 20 30 40
50 60 70 80


radife chand , sotoone chand 
ba y adad nmishe hjasho tashkhis dad
do adad mishe


radife 1 , sotoone2
'''


my_array = np.array([ [10,20,30,40], [50,60,70,80]])

#zarf[radif,soton]
my_array[1] #array([50, 60, 70, 80])

my_array[1,2] #70



#------------------------------
'''
Matplotlib --> ketabkhon dg --> rasme
koli rasm

bad az gpt kasi in ktabkhoen haro yad nmigire
khode gpt khoshgel va herfe ee brt misaze

shoma bayd codesho bfhmi k cho b chie





#--->installl

pip install matplotlib
pip install matplotlib.pyplot


https://matplotlib.org


shoro --> https://matplotlib.org/stable/index.html


examples-->
https://matplotlib.org/stable/plot_types/index.html



'''



import matplotlib.pyplot as plt

x= [1,2,3,4,5]
y= [10,20,30,40,50]

plt.plot(x,y)
plt.show()






import numpy as np


x = np.linspace(0, np.pi , 200)
y = np.sin(x)


import matplotlib.pyplot as plt

plt.plot(x,y)
plt.title('my fucntion')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()





#-----------
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, np.pi, 200)
y = np.sin(x)

# Create the figure
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(x, y, linewidth=2)

# Axis labels and title (LaTeX formatting)
plt.xlabel(r"$x$", fontsize=12)
plt.ylabel(r"$\sin(x)$", fontsize=12)
plt.title(r"Sine Function: $y = \sin(x)$", fontsize=13)

# Grid and layout for publication quality
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Display the plot
plt.show()



#------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------- Global style configuration --------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "mathtext.fontset": "stix",
})

# -------------------- Data generation --------------------
x = np.linspace(0, np.pi, 200)
y = np.sin(x)

# -------------------- Figure creation --------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=300)

ax.plot(
    x,
    y,
    linewidth=2.0,
)

# -------------------- Axis formatting --------------------
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\sin(x)$")
ax.set_title(r"Sine Function: $y = \sin(x)$", pad=10)

ax.set_xlim(0, np.pi)
ax.set_ylim(-1.05, 1.05)

ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

# Improve tick appearance
ax.tick_params(direction="in", length=5, width=1)

# Remove unnecessary whitespace
fig.tight_layout()

# -------------------- Display / Save --------------------
plt.show()



#-plot()
#scatter()

#--> 


