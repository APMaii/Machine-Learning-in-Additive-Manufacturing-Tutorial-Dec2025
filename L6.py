"""
In The Name of GOD

Dec 20 2025

@author: Ali Pilehvar Meibody


L6





--> L5 --> review on Python ()
alipilehvar1999@gmail.com


--> L6 --> 

script --> Ampro.py --> saved bashe jaee

import Ampro


Ampro.fucntion()



from Ampro import function


function()




from Ampro import function as f

f()


#-->scrip benevisim
ketabkhone hastan koli script ,koli tabe , class ,.....
baraye hite haye mokhtalaf

--> Pypi 
--> Github





#------ by default (niazi b install nist)
random
time
math



"""

#yekbar balaye hameye code ha k dg python beshnasatesh
import math

math.sqrt(100) # 10.0

zarf = math.pi
math.sin(zarf)

math.cos(zarf)


#dar jalseyegozashte --> listi az tabe haye math


import random

random.randint(10,20) #15

random.uniform(10,20) #10.191108752812303



import time

time.timezone #-3600

time.time() # 1766243233.498879



'''
Python ye samt --> yek fardi --> Guido van Rossum
organization  ---> yseri afrad ketabkhone nevshtn 

Numpy --> 2006  --> calculation
-->rooye numpy scipy , sympy --> numerical modeling 

Matplotlib --> rasme nemodoar
Pandas --> kar ba dade, cleaning data


#maroof darimd ar hame hite
#opencv --> computer vision 

pygamlab --> 3,4 sal gozashte , 20K 


computere shoam ianro ndre, numpy , pandas , 
install konid


--> varede environement



Anaconda ---->  [lapotob] --> chanta enviuroenmnt -->chanta laptab drid

anaconda navigator ro baz mikonid [environment] --> ye laptabe jadid bsazi (enviroenemnte jadid) --.esme -->
roo in lapotob ch vrsioni az python bashe


[Home] -->   listi envrionemntat hast 

anaconda -->mohit sakhtam --> ye applicartion spyder , ...

search spyder --> spyder (base koel laptab) , spyder ( additive)  , spyder(Profsaboori)


package --> ha ham bayad bri too mohite khdoet oonja downlaodesh kon (installesh koni)

numpy ro nasb kon computeret bshanse --> too koel computer brizi

3.10.2   -->numpy , .... ---> spydere hamon mohiteto baz koni


MAC ---> Terminal --> safe siahe (shell)


Windows ---> win + R  wt --> conda --> error mide conda ro nmishansam
search bar samte chap paeen --> anaconda --> [anaconda prompt] --> siah

---> y safe meshki

(base) apm@APMs-MacBook-Pro ~ % 

--> (base) -->koel alptabam


ychizi nasb koni --> bayd bri too oon mohitet 


(base) apm@APMs-MacBook-Pro ~ % conda activate DL
(DL) apm@APMs-MacBook-Pro ~ % 

karato koni
bekhay biay biron -->

(DL) apm@APMs-MacBook-Pro ~ % conda deactivate
(base) apm@APMs-MacBook-Pro ~ % 




#---->miri dakhe;
(base) apm@APMs-MacBook-Pro ~ % conda activate DL
(DL) apm@APMs-MacBook-Pro ~ % pip install numpy
Requirement already satisfied: numpy in ./anaconda3/envs/DL/lib/python3.10/site-packages (1.26.4)




pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn


#--optional
pip install PyGamLab

--> 4 ta ketabkhone nasb mishe

-->esme ketabkhone ro search bezan + pip -->safeye
Pypi.org --> 



ejaze mide


too ye mohiti inaro rikhti --> mohite DL

spuydere DL 


too ye mohtie dg , spyder khali --> import numpy --> error 


-->mokhafaf daran

import numpy as np
import pandas as pd
import matoplotlib.pyplot as plt


'''


#--------Jalase gozashte sohbat shod

#-->List 
#roo kernel python, 1 bodi bood
a= [10,20,30,40]


#numpy --> array --> 60x faster , chan bodi besazi

import numpy

my_arr= numpy.array([10,20,30,40])


print(type(my_arr)) #<class 'numpy.ndarray'>

my_arr.ndim #1
my_arr.size #4
my_arr.shape #(4,)

import numpy as np
#radifa(takid mikonm)
#soton

my_2arr = np.array([ [10,20,30,40] , [50,60,70,80]   ])
 

my_2arr.ndim #: 2
print(my_2arr)

'''
[[10 20 30 40]
 [50 60 70 80]]

'''

#zarf[radif] -->kole radfifo
my_2arr[0] #array([10, 20, 30, 40])
my_2arr[1] #array([50, 60, 70, 80])

#zarf[radif,soton] -->element

#70-->radife 1 , sotone 2
my_2arr[1,2] #70

#sotone 2 --> hey avaz koni radifa
my_2arr[0,2] #30
my_2arr[1,2] #70

my_2arr[ :, 2] #array([30, 70])

#np.sin()
#np.cos()

#np.log()

#np.mean()
#np.max()
#np.min()
#np.sum()
#np.

#-->toye mohasebat komak mikone easy

'''
numpy --> numpy.org --->
documentation 


--> [getting_started ,quickstart]
-->[API Refercence , tutorial]


done doen step step mitonid hamasho az bar bshid

yechizio az bar bshid


ye function
API rference --> search XXXX chatgpt --> az gpt rajebe y function soal beporse



'''
#----------------------------
'''

Matplotlib ---> In baraye plot ha o rasm ha hast 

ye ketabkhone -->
XX --> tya ghabl az in yek --> 4 --> 10 ta jalase matplotlib dars midadam 

emroze gpt --> mitone yek plot haee rasm kone k ensan b hich onvan nmitone , Senior (8,9 sal matplotlib)

gpt ham inghdam easy tolid nmikone 

#--> beshnasim k matplotlib chijori tolid mikone
#--->ba yeseri tavabe o structuresh ashan bashim (geenrate kard , ychizesho tagir bde , injasho taghir)
#--->journal ha  ,article --> x, y adadasho ghati bzne , y ,x  --> shoam t avaghty ndoni
plotlib chikar mikonii easy az gpt estefade koni


recommend mdiam k az gpt estefade kon ama ama bayad bdoni matplotlib chikar mikone k betooni supervise koni (nezarat koni)
plot --> khoshgel , professional , Accurate bashe




conda activate env
pip install matplotlib
pip install matplotlib.pyplot
conda deactivate


import matplotlib

import matplotlib.pyplot


import matplotlib.pyplot as plt

plt.plot()
plt.scatter()
plt.hist()
plt.pie()
.....



https://matplotlib.org/stable/plot_types/index.html





'''

import matplotlib.pyplot as plt

#x,y , ........ mitoni bdi 

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,20,30,40,50,60,70,80,90,100]

#10 ta noghte daram (x,y) --> (1,10) (2,20) (3,30) x -->list, y correspond -->list


#step 1-->besazi
plt.plot(x,y)

#step2 --> ziba sazi


#step 3 --> neshonesh bde
plt.show()



#-->spyder easy mitone done done run koni

#in L6.py -->github

#.py

#.ipynb --> Jupyter notebook (spyder) --> code ha 
#ejra shode toosh abshan -->tadris, presentation awlie


#=====================


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,np.pi , 200)
y = np.sin(x)


x_2 = np.linspace(0,np.pi , 50)
y_2 = np.cos(x_2)

#scatter, plot --> label
plt.plot(x,y,label='sin(x)')
plt.scatter(x_2,y_2,s=5, label='cos(x)')
plt.legend()
plt.show()


#============================
'''
Pandas --> pish daramadi ML 

list()
np.array()

plt.plot()




-->datahamo -_.dasti

np.array???

chizi hast k bman komak kone datahamo biaram too python?



PANDAS --> inakro rahat krde


'''

#---.python list
a= list([10,20,30,40])
#slow() , 1D


#-----Numpy array()
b = np.array([[10,20,30,40] , [50,60,70,80]])

#fast , 2D ->jadval 
#khodet manually--> np.arange np.linsopace np.random ,np.zero

#numpy array

#excel, csv ,...
#----pandas DataFrame --> in kari k shoma mikhayd do halat
#dasti dataframe (array)
#pandas dataframe

#   tavabe dare excel, csv shomar migire --> dataframe (python)

#data ha bshim 

#dasti

import pandas as pd

#pd --> mokhafafe pandas hast

#DataFrame


d1 = pd.DataFrame(data=[10,20,30,40])


print(type(d1)) #<class 'pandas.core.frame.DataFrame'>

#list, np array , pd dataframe (df)


#list
a = [10,20,30,40]
b= np.array([10,20,30,40])

#numpuy array --> fast

c = pd.DataFrame(data=[10,20,30,40])


#ma mitonim baray yek data --> label bzarim 



b = np.array([[10,20,30,40],[50,60,70,80]])


c = pd.DataFrame(data = [[10,20,30,40],[50,60,70,80]])
c = pd.DataFrame( [[10,20,30,40],[50,60,70,80]])



#list, numpy
c = pd.DataFrame(data = [[10,20,30,40],[50,60,70,80]], columns=['sample','speed','temp','modulus'])


#inkar anjam nmishe ha

c = pd.DataFrame(data = [[10,20,30,40],[50,60,70,80]] , columns = ['sample','speed','temp','modulus'] , index =['sample1','sample2'])






#-------

'''
10 20 30 40
50 60 70 80
'''


b = np.array([[10,20,30,40],[50,60,70,80]])

c = pd.DataFrame( [[10,20,30,40],[50,60,70,80]])

b[0] # array([10, 20, 30, 40])


c[0]

'''
Out[36]: 
0    10
1    50
Name: 0, dtype: int64
'''

c[1] #soton

#list, --> sotono azam migir


#soton, radif
#c[2,1] element o .. yejor dg behet migm

c = pd.DataFrame( [[10,20,30,40],[50,60,70,80]] , columns = ['temp','speed','pressure','modulus'])


'''
temp speed pres modulus
10    20   30   40
50    60   70   80
'''

#index --> esm gozari shdoe

#dataframe --> esmi nbshe c[0] -->sotone 0
#ag esm dashte bashe


c['temp']

'''
0    10
1    50
Name: temp, dtype: int64

'''


c['speed']

'''
0    20
1    60
Name: speed, dtype: int64

'''



c[['speed','temp']]

'''
   speed  temp
0     20    10
1     60    50

'''



c.loc[0]

'''
temp        10
speed       20
pressure    30
modulus     40
Name: 0, dtype: int64

'''

c.loc[1]

'''
temp        50
speed       60
pressure    70
modulus     80
Name: 1, dtype: int64

'''

'''
temp speed pres modulus
10    20   30   40
50    60   70   80
'''

#soton --> zarf[esme_soton]
#radif --> ag radif ha esm --> zarf.loc[]   || adad dasht --> zarf.iloc[]

#y eleemnt brsm?
#soton , radif

c['temp'].loc[0] #10

c['speed'].loc[1] #60

c['speed']

'''
0    20
1    60
Name: speed, dtype: int64

'''

#----> manually salhti

#numpy --> pandas dataframe


#numpy --> 


import numpy as np
np_arr= np.arange(0,200) #[0,.....20]

#range , linspace ,.....

np_arr2 = np_arr.reshape(50,4) #50 radif, 4 ta soton



import pandas as pd


df = pd.DataFrame(data = np_arr2)


df = pd.DataFrame(data = np_arr2 , columns = ['sample','speed','temp','solubility'])









#---> dataye khodamo chijori biaram??

#pandas -->pd

#pd.DataFrame() ---> datframe bsazi


#pd.read_excel()
#pd.read_csv()
#pd.red_json()



'''
---> .format

ax --> .jpg  .png 
video --> .mp4 .mov .gif

Data --> .xlsx .xls .csv  .db .json ,.....


file haee k b formate --> .xlsx .xls .csv  .db .json ,.....

pandas -> tabe e dare kk mitone az birone laptabet datato begire
va chiakr kone?? --> tabdil koen b dataframe --> type k python mishnase


'''


#ag fileton .csv --> csv hast
pd.read_csv()


#ag fileton .xlsx .xls --> 
pd.read_excel()



#pd.read_felan(path = addrese locatione oonjae k filete behem bgo brm biaramesh)

#chegone directory ro da rbairim
#------->

#/Users/apm/Desktop

#C://Users/apm/desktop/ddd/dddd


#/Users/apm/Desktop/ESMEFIEL BA FORMAT

#/Users/apm/Desktop/data.csv




#csv ro mire migrde donbael in location , dataro bar midare
#save mikone tooye zarf 

zarf = pd.read_csv('/Users/apm/Desktop/data.csv')



#/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx


zarf2 = pd.read_excel('/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx')




'''

pandas --> np.array --> DataFrame --> label column, row 

--> dasti bsazi 
zarf = pd.Dataframe(data= , columns=[] , index = [])

zarf = pd.read_felan('addres/ / / / /data.format')
.xlsx .xls --> read_excel('......')
.csv --> read_csv('.....')



zarf --> dataframe




'''
zarf2 = pd.read_excel('/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx')



print(type(zarf2)) #<class 'pandas.core.frame.DataFrame'>


#zarf , esmi k gozashiu
zarf2.columns






df = pd.read_excel('/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx')


df.columns


'''
Index(['Drug', 'Mw', 'Melting point', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility'],
      dtype='object')

'''


df.columns = ['Drug name', 'Mw', 'Melting point', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility']




df.head()

'''
Out[63]: 
      Drug name      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0  clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1  clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2  clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3  clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4  clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04

[5 rows x 10 columns]

'''



df.head(10)

'''
      Drug name      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0  clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1  clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2  clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3  clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4  clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04
5  clotrimazole  344.84          421.0  ...  17.8              0.5  5.280000e-04
6  clotrimazole  344.84          421.0  ...  17.8              0.6  1.410000e-03
7  clotrimazole  344.84          421.0  ...  17.8              0.7  4.060000e-03
8  clotrimazole  344.84          421.0  ...  17.8              0.8  7.200000e-03
9  clotrimazole  344.84          421.0  ...  17.8              0.9  1.200000e-02

[10 rows x 10 columns]

'''




df.tail()

'''
Out[65]: 
           Drug name      Mw  Melting point  ...  TPSA  EtOH mass ratio  solubility
970       Etoricoxib  358.80          407.5  ...  68.3              0.9     0.06137
971       Etoricoxib  358.80          407.5  ...  68.3              1.0     0.05374
972  beta-Artemether  298.37          361.0  ...  46.2              1.0     0.04630
973  beta-Artemether  298.37          361.0  ...  46.2              0.8     0.00866
974  beta-Artemether  298.37          361.0  ...  46.2              0.6     0.00135

[5 rows x 10 columns]

'''



df.tail(7)

'''
           Drug name      Mw  Melting point  ...  TPSA  EtOH mass ratio  solubility
968       Etoricoxib  358.80          407.5  ...  68.3              0.7     0.01520
969       Etoricoxib  358.80          407.5  ...  68.3              0.8     0.03616
970       Etoricoxib  358.80          407.5  ...  68.3              0.9     0.06137
971       Etoricoxib  358.80          407.5  ...  68.3              1.0     0.05374
972  beta-Artemether  298.37          361.0  ...  46.2              1.0     0.04630
973  beta-Artemether  298.37          361.0  ...  46.2              0.8     0.00866
974  beta-Artemether  298.37          361.0  ...  46.2              0.6     0.00135

[7 rows x 10 columns]

'''



df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 975 entries, 0 to 974
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Drug name        975 non-null    object 
 1   Mw               975 non-null    float64
 2   Melting point    975 non-null    float64
 3   density          975 non-null    float64
 4   XLogP3           975 non-null    float64
 5   HBD              975 non-null    int64  
 6   HBA              975 non-null    int64  
 7   TPSA             975 non-null    float64
 8   EtOH mass ratio  975 non-null    float64
 9   solubility       975 non-null    float64
dtypes: float64(7), int64(2), object(1)
memory usage: 76.3+ KB

'''


df.describe()

'''
               Mw  Melting point  ...  EtOH mass ratio  solubility
count  975.000000     975.000000  ...       975.000000  975.000000
mean   308.784413     467.797056  ...         0.508332    0.006459
std    117.419091      70.519339  ...         0.315625    0.021161
min    130.080000     344.000000  ...         0.000000    0.000000
25%    211.220000     413.000000  ...         0.200735    0.000040
50%    290.320000     459.000000  ...         0.500000    0.000373
75%    362.470000     512.000000  ...         0.800000    0.003220
max    639.820000     623.000000  ...         1.000000    0.241000

[8 rows x 9 columns]

'''


df.column = ['Drug', 'Mw', 'Melting point (K)', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility']

#data analysis anjam midi

#--------


#pandas + matplotlib y poli bznm


#plt.scatter(x,y)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx')


df.columns
'''
Index(['Drug', 'Mw', 'Melting point', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility'],
      dtype='object')


solubility bar asase melting point rasm kone

x = melting point | ssolutbity --> y

'''

x = df['Melting point']
y = df['solubility']

plt.scatter(x,y , s=5)
plt.ylabel('Solubility')
plt.xlabel('Melting point')
plt.grid()
plt.show()


#-------------------------------


x = df['Melting point']
y = df['solubility']

plt.scatter(x,y , s=5)
plt.ylabel('Solubility')
plt.xlabel('Melting point (K)')
plt.grid()
plt.show()


'''

zahmat bekeshid --> data haee k darid 
az ghabl ya baraye thesis , ya abraye magahel

yekbar --> pandas.read_excel() read_csv --> zarf

.columns -->soton

.head()

.tail()

.info()

.describe()



x = data[sotone roed
y = 


matplotlib 

plt.scatter(x,y)



plt.hist()



#----->



Nimsaate aval --> mohem trin bakhsh haye pandas ro baraye cleaninge data pish mibrim


data analysis --> bahse kh piuchide hast , filter koni data , ... inkaro onkaro koni --> Documentation mojod hastan linkesho azizan mzirm

jalase --> Cleaning data -->datamon clean 

data --> miad varede python --->mosheklat ya khdoesh dahste , pish baid conversion -->source error -->chian

detect , clean --> amadeye raftan b Machien elarning




L7 --->
Linear regression --> sklearn.linear regression 

linear regression --> Machine elarnin (DT , RF , KNN)

2 hours


L8 -->
(SVR , MLP (artificial neural network))
Cross validation (raveshi)
gridsearch (hyperparameter tuning)

Case study moarefi koni , 2 nfr 
bian va bahm kareshono codi barashon L8 ROOYE DATA On afrad bznim

chijori dakhele maghale mishe in haro gozaresh krd --> yek maghale ee hast 

charchoob anjam

L8 -->3 Hour



'''











