#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 15:58:42 2025

@author: apm



L7-----------

Pandas completed
Machine learning theory  --> Kh moheme




L8---------
Applied machine learning in Additive manufacturing 



"""


'''

Part1 - Python Basic -------------
1-python built in fucntions
2-Keywords (if , else, elif , for , while , def , class)
3-Variables (numbers, str, list, tuple , dictioanry ,...)



Part2 - Libraries---------------
import library
from library import fucntion
from library import function as fun
Math , time, random


Part 3 - Advanced Libraries----------
Numpy -->np.array() -->faster, 2d data
Matplotlib --> visualization 
Pandas --> pd.dataframe --> excel, csv ,.. column label 





Part4 - Machine learning------
Theory 
regression
Classification
Sklearn --> scikit-learn --> ketyabkhane hast


'''


#----------List-------

a=[10,20,30,40,50]
a=list([10,20,30,40,50])

print(type(a)) #<class 'list'>

#esm[index]
a[0] # 10
a[0:3] #[10, 20, 30]

a[0] =13090
print(a) #[13090, 20, 30, 40, 50]

#list functions
a.append(1000)
print(a) #[13090, 20, 30, 40, 50, 1000]

a.remove(1000)
print(a) #[13090, 20, 30, 40, 50]



#--Mohasebat --> Python , C++ Fast ,  2D variable
#row , columns --> Numpy -->Number + python

#pip install numpy --> nasb beshe in ketabkhone
#kole ketabkhone ro impor tmikonim va mokhafaefsahm np
import numpy as np


#np--> np.array() jaye list ro migrie baray ema

b= np.array([10,20,30,40,50])

print(b) #[10 20 30 40 50]

print(type(b)) #<class 'numpy.ndarray'> --> C++ kernel hasf -_>sri tar

b[0] # 10

b[0:3] # array([10, 20, 30])

b[0] = 13090

print(b) #[13090    20    30    40    50]

#tabe hash inghd sade nis mese list ha 
#sin , cos , ........ bahash bokoni

b2 = np.array([ [10,20,30,40] , [50,60,70,80] ])
print(type(b2)) #<class 'numpy.ndarray'>
print(b2)
'''
[[10 20 30 40]
 [50 60 70 80]]

'''
b2.ndim #2
b2.size #8
b2.shape # (2, 4)
#2 ta radif, 4 ta soton

#numpy --> radif mohem tare

b2[0] #radife 9o  # array([10, 20, 30, 40])

#radife chandom , sotone chandom

b2[1,2] # 70

b2[1,2] =10000000

print(b2)
'''
[[      10       20       30       40]
 [      50       60 10000000       80]]

'''


#mohemtarin tavabe e k bodo --> generator

d=np.arange(0,100)


e = d.reshape(2,50)

f = d.reshape(50,2)

a = np.random.randint(0,100,(10,))

b = np.random.randint(0,100,(30,2))


x = b

y = np.sin(x)

np.mean(y)

np.sum(y)





#--------- Matplotlib

import matplotlib.pyplot as plt
x=np.linspace(0,2*np.pi,200)

y= np.cos(x)

plt.plot(x,y)
plt.title('samples')
plt.xlabel('Tetha')
plt.ylabel('Cos()')
plt.show()



#plot() , scatter() , bar() , hist() ,....

#noghte
plt.scatter(x,y)
plt.title('samples')
plt.xlabel('Tetha')
plt.ylabel('Cos()')
plt.show()



#to harkodom az in tabe ha to oparnatez taghirati ija djkoni

plt.scatter(x,y,s=1)
plt.title('samples')
plt.xlabel('Tetha')
plt.ylabel('Cos()')
plt.show()



plt.scatter(x,y,s=1,c='r')
plt.title('samples')
plt.xlabel('Tetha')
plt.ylabel('Cos()')
plt.show()




#------------------------
#Numpy, Matplotlib, Pandas 

#ghabl az inke b datahye khdoet fek koni yadet bashe
#ye ki has bname pandas -->pd 

import pandas as pd


#man yechi daram bname DataFrame --> numpya rray 
#sootona , radifa mitoni label  bzari --> ahan in yekam real


'''


10  400
20  420
30  410
40  390


Np array

  0  1
0 10 400
1 20 420 
2 30 410 
3 40 390

array[0]
array[:,0]


pd DataFrame

  temp  modulus
0 10    400
1 20    420 
2 30   410 
3 40   390


     temp  modulus
case1 10    400
case2 20    420 
case3 30   410 
case4  40   390







'''


import pandas as pd


ndarray = np.array([[10,400],[20,390],[30,370],[40,350]])
pandas_df = pd.DataFrame([[10,400],[20,390],[30,370],[40,350]] , columns =['temp','modulus'])


'''

MY dataframe
   temp modulus
0  10   400
1  20   390
2  30    370
3  40   350



'''
#soton moheme bad radif moheme --> baraxe numpy



pandas_df['temp']
'''
0    10
1    20
2    30
3    40

'''


pandas_df['modulus']





#loc , iloc

#esm daran --> loc
#age adadi --> iloc  --> i(index adadie)

pandas_df.iloc[0]
'''
temp        10
modulus    400

'''

pandas_df.iloc[1]

'''
temp        20
modulus    390

'''


#---numpy yag yadet abshe

#zarf[koodmrafid , kodom sotoon]

#column, row , element???

#column , row

#zarf[soton].iloc[radif]

pandas_df['temp'].iloc[0] # 10


pandas_df['temp'].iloc[0] = 8

print(pandas_df)

'''
   temp  modulus
0     8      400
1    20      390
2    30      370
3    40      350

'''


#numpy array --> more fast than Pandas dataframe



#Pandas DataFrame --> yek sotoni chizi ro az excel o .,..



#file --> .xlsx .xls --> excel || .csv --> csv

#masir -->  /user/apm/folder/desktop/folde/data.format

#path --> /user/apm/folder/desktop/folde/data.format -- >vorodi



#xlsx xls ---> pd.read_excel(path)
#csv -->      pd.read_csv(path) 


#jadvali k darid ro tabdil mikone be yek DataFrame



zarf = pd.read_excel('/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx')


zarf.columns


'''
Index(['Drug', 'Mw', 'Melting point', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility'],
      dtype='object')

'''

zarf['Drug']


'''
Out[53]: 
0         clotrimazole
1         clotrimazole
2         clotrimazole
3         clotrimazole
4         clotrimazole
      
970         Etoricoxib
971         Etoricoxib
972    beta-Artemether
973    beta-Artemether
974    beta-Artemether
Name: Drug, Length: 975, dtype: object
'''


drug_column = zarf['Drug']


#973 ovim drugesh bbini xchie

#kodom soton, kodom radif


zarf['Drug'].iloc[973]

#'beta-Artemether'


zarf['Mw'].iloc[973] # 298.37


#kolan bman in daro 973

#soton

zarf['Mw']



#radif

zarf.iloc[973]

'''
Drug               beta-Artemether
Mw                          298.37
Melting point                361.0
density                     1.0733
XLogP3                         3.1
HBD                              0
HBA                              5
TPSA                          46.2
EtOH mass ratio                0.8
solubility                 0.00866
Name: 973, dtype: object

'''



#elemeent -->soton, radif

zarf['Drug'].iloc[910] # 'Metronidazole'


#------------ data oomade 

#ghablaz inke varede 

#pandas --> dataframe --.zarfet

#khoet besazi

zarf = pd.DataFrame([10,20,30,40],column=['temp'])

#-->dataro biaran amade
zarf2 = pd.read_excel('/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx')


#jofteshonnn variable -->Dataframe --> Pandas hastan

#class --> methods (tabe hast)

zarf.abs()
zarf.add()
zarf.div()
zarf.max()

zarf.min()
zarf.argmax()
zarf.argmin()
zarf.copy()
zarf.isin()
zarf.isna()
zarf.isnull()
zarf.dropna()
zarf.drop()
zarf.fillna()


#------------

values = [ [ 1985 , np.nan , 'Biking', 68] ,
          [1984 , 3 , 'Dancing' , 83] ,
          [1992, 0 , np.nan , 112]]




df = pd.DataFrame(values, columns=['birthday', 'children','hobby','weight'],index = ['alice','bob','charles'])
#excdl omde ya harchi
#name
#listt
#my_list
#df -->dataframe
#df1 , df2
#data



#---filtering


#kasanik kochiktar az 1990

#yek sotonet az yekseri chiza kmtar bashe


#tmperature <80 

#speed >100

#zarf[dakhele in shart mizari]

df['birthday']

'''
alice      1985
bob        1984
charles    1992
Name: birthday, dtype: int64
mitoni bbini, zakhirash koni

'''



#aya bzoorg tar az 1990?
df['birthday']>1990
'''
alice      False
bob        False
charles     True
Name: birthday, dtype: bool


True false pas mdie
'''




#beman True haro bde
df[ df['birthday']>1990 ]

'''
         birthday  children hobby  weight
charles      1992       0.0   NaN     112

'''

#filter krdm

bigger_1990_lists = df[ df['birthday']>1990 ]



#filterin mikonm


#data


#very_hot_sampels = data[ data['temp']>2000 ]



#adding columns -->

df['age'] = 2025 - df['birthday']

#pas nmide -->roye df taghir mdie


#filtering balaye 40 salan ro

df['age']
'''
alice      40
bob        41
charles    33
Name: age, dtype: int64

'''

df['age']>39
'''
alice       True
bob         True
charles    False
Name: age, dtype: bool

'''



df[ df['age']>39 ]

'''
       birthday  children    hobby  weight  age
alice      1985       NaN   Biking      68   40
bob        1984       3.0  Dancing      83   41

'''


zarf_jadid = df[ df['age']>39 ]




#----filter, add , remove

#soton, radif 
#tabe ee bvvojod --> drop()
import numpy as np
import pandas as pd
a= np.random.uniform(0,10,size=(50,3))

#50 ta radif, 3 ta sotoon
#numpy -->pandas
#list -->pandas
data = pd.DataFrame(a,columns=['temp','speed','modulus'])


#alan manually neevshtam

#hamchind ataee dari excel , ...

data = pd.read_excel('user/apm/desktop/...../data.format')
data = pd.read_csv('user/apm/desktop/...../data.format')



data = pd.DataFrame(a,columns=['temp','speed','modulus'])


#hamashon--> DataFrame -->yekseri adad daan, esme soton, index daran row

#yek soton hazf konm


new_data = data.drop(columns='modulus')


new_data2= data.drop(index=3)




#--- khrooji mide
#hey nemikhay done done zarf bsazi

#---> pandas too dele tamame tavabe ash --> inplace

#inplace = False

#inplace = True -->khroji nemide , mostaghim emal mione

#bejaye inkar
new_data = data.drop(columns='modulus')



data.drop(columns='modulus',inplace=True)


data.drop(index=8 , inplace=True)


data.reset_index(drop=True,inplace=True)

#drop = False --> index haye gahblio-->sotone jadid mikone mizare kenare dastet 






#------ chanta pandas dataframe ro bechabsonid beham

df1= pd.DataFrame([1,2])

df2 = pd.DataFrame([3,4])


df3 = pd.concat([df1,df2])

print(df3)
'''

1
2
3
4

pd.concat()
pd.vstack()
pd.hstack()
pd.zstack()

---> Pandas Documentation ... Full 
'''


print(type(zarf)) #<class 'pandas.core.frame.DataFrame'>


zarf.columns
'''
Index(['Drug', 'Mw', 'Melting point', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility'],
      dtype='object')

'''


zarf.head()


'''
           Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0  clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1  clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2  clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3  clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4  clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04

[5 rows x 10 columns]

'''



zarf.tail()

'''
                Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio  solubility
970       Etoricoxib  358.80          407.5  ...  68.3              0.9     0.06137
971       Etoricoxib  358.80          407.5  ...  68.3              1.0     0.05374
972  beta-Artemether  298.37          361.0  ...  46.2              1.0     0.04630
973  beta-Artemether  298.37          361.0  ...  46.2              0.8     0.00866
974  beta-Artemether  298.37          361.0  ...  46.2              0.6     0.00135

[5 rows x 10 columns]

'''




x = zarf['Melting point'] #datframe zarf sotone ino vrdar

y = zarf['solubility'] #dataframe zarf sotoen solubility



print(len(x)) #975

#975 ta data drm , 075 x 975 y

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.title('my drug data')
plt.xlabel('Tm')
plt.ylabel('solubility')
plt.show()



#zarf --> soton bekshid biron bairidf az matplotlib az
#line, scatter , bar , ...

zarf.plot(kind='scatter',x='Melting point' , y='solubility')
plt.show()


#dele pandas -->matplotloib.pyplot() ezafe krdn
#k shoam fght ba yadgirie pandas hata btoni tamame inakraro bokoni


#khondan??

#read_

pd.read_csv('path')
pd.read_excel('path')


#filtrng, harkari bode kardi (agar taghir dadi
#drop ,... 


#karet tamom shod -->save koni

zarf.to_csv('C/user/...../esme_file.csv')

zarf.to_excel('c/user/..../esme_file.xlsx')





#========================
'''
Ghabl azinke varede Ml beshim

step 0 --> Cleaning 


Data ro tamzi konim momkene az excel, csv k darim
ta dataframe bshe --> yekseri moshkelat pish biad

va mosheklato 1-tashkhis bdim 2-clan kon(okeyesh konim) hazfesh, jaygzoari ,....





'''

#path ---> masirsho dar miarid 
#proeprties , get info --> dirdctory
#/users/apm/desktop/....../esme_file.format

path = '/Users/apm/Desktop/MASTER THESIS/drug_data.xlsx'

#xlsx_ xls

data = pd.read_excel(path)


#csv
data = pd.read_csv(path)


print(type(data)) #<class 'pandas.core.frame.DataFrame'>




data.head()
'''
           Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0  clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1  clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2  clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3  clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4  clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04

[5 rows x 10 columns]

'''


data.head(10)

'''
           Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
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


data.tail()
'''
                Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio  solubility
970       Etoricoxib  358.80          407.5  ...  68.3              0.9     0.06137
971       Etoricoxib  358.80          407.5  ...  68.3              1.0     0.05374
972  beta-Artemether  298.37          361.0  ...  46.2              1.0     0.04630
973  beta-Artemether  298.37          361.0  ...  46.2              0.8     0.00866
974  beta-Artemether  298.37          361.0  ...  46.2              0.6     0.00135

[5 rows x 10 columns]

'''



data.tail(8)

'''
                Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio  solubility
967       Etoricoxib  358.80          407.5  ...  68.3              0.6     0.00459
968       Etoricoxib  358.80          407.5  ...  68.3              0.7     0.01520
969       Etoricoxib  358.80          407.5  ...  68.3              0.8     0.03616
970       Etoricoxib  358.80          407.5  ...  68.3              0.9     0.06137
971       Etoricoxib  358.80          407.5  ...  68.3              1.0     0.05374
972  beta-Artemether  298.37          361.0  ...  46.2              1.0     0.04630
973  beta-Artemether  298.37          361.0  ...  46.2              0.8     0.00866
974  beta-Artemether  298.37          361.0  ...  46.2              0.6     0.00135

[8 rows x 10 columns]

'''


data.describe()

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


data.columns

'''
Index(['Drug', 'Mw', 'Melting point', 'density', 'XLogP3', 'HBD', 'HBA',
       'TPSA', 'EtOH mass ratio', 'solubility'],
      dtype='object')

'''

data['Mw'].describe()
'''
count    975.000000
mean     308.784413
std      117.419091
min      130.080000
25%      211.220000
50%      290.320000
75%      362.470000
max      639.820000
Name: Mw, dtype: float64

'''

#soton b sotone mirim mibinim

data['Mw'].mean() #308.7844133333333

data['Mw'].max() # 639.82



#rasm bashe
data.plot(kind='scatter', x='esme soton' ,y='esme tono')
plt.show()









data.info()
'''

# type et chie --> DataFrame bayad bashe
<class 'pandas.core.frame.DataFrame'>

#chanta radif dari --> 975 ta radif dri 975 nemoone
#30 , 40 ta nemone , moshakhasat --> column soton

row , radif, entry --> tedade nemonat index 0  1 2 3
Column , property , sotoon --> temp , ...



RangeIndex: 975 entries, 0 to 974

#10 ta soton dare va soton haro 
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Drug             975 non-null    object 
 1   Mw               975 non-null    float64
 2   Melting point    975 non-null    float64
 3   density          975 non-null    float64
 4   XLogP3           975 non-null    float64
 5   HBD              975 non-null    int64  
 6   HBA              975 non-null    int64  
 7   TPSA             975 non-null    float64
 8   EtOH mass ratio  975 non-null    float64
 9   solubility       975 non-null    float64
 
 baraye har sotoon 
 
 975 tae k dari 975 ta non-null hast 
 
 
 null-->khali --> Empty cell --> shayad y akhodet khali gzoashti
 ya vaghty ovordish khali shode 
 
 
 
 
dtypes: float64(7), int64(2), object(1)
memory usage: 76.3+ KB

'''



'''

4 ta no eshkal tarafim
data = pd.read_....

data --> Pandas dataframe


1- Empty cells --> kahlian, computeret
data.info() ---> har soton non null = len(datat)


'''


data = np.array([[10,20],[30,40] ,[None,50],[60,80]])


df = pd.DataFrame(data,columns=['temp','modulus'])


print(df)


'''
   temp modulus
0    10      20
1    30      40
2  None      50
3    60      80


900 ta data 

'''


df.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   temp     3 non-null      object
 1   modulus  4 non-null      object
dtypes: object(2)
memory usage: 192.0+ bytes




1 done null --> empty dari -_> injori msihe fahmid

'''

df.isnull()

'''
    temp  modulus
0  False    False
1  False    False
2   True    False
3  False    False


400 ta data dashte bashdi 
true fasle
'''


df.isnull().sum()

'''
temp       1
modulus    0
dtype: int64

'''


zarf.isnull().sum()

'''
Drug               0
Mw                 0
Melting point      0
density            0
XLogP3             0
HBD                0
HBA                0
TPSA               0
EtOH mass ratio    0
solubility         0
dtype: int64

'''




#khodet khalish gozashti ya inke khalishode


#---> remove konish -->datye kafi dari
#mitoni beri remove koni easy

#df.dropna() maid datato na drop mikone --> dataframe jadid


#emal bshe -->  tavabe -> Inpalce = True


df.dropna(inplace=True)


#data dari mitoni hazf koni easy



#fill koni


data = np.array([[10,20],[30,40] ,[None,50],[60,80]])
df = pd.DataFrame(data,columns=['temp','modulus'])

df.fillna(10,inplace=True) #20,30,....




temp_mean = df['temp'].mean() #27.5


df.fillna(temp_mean , inplace=True)


#--kheyli dari

df.fillna(method = 'ffill',inplace=True) #ghablisisho varmidari

df.fillna(method='bfill' , inplace=True) #badisho var mdiare




'''

4 ta no eshkal tarafim
data = pd.read_....

data --> Pandas dataframe


1- Empty cells --> kahlian, computeret
data.info() ---> har soton non null = len(datat)
data.isnull().sum()
data.dropna() tamame onae k kahlian miparan
data.fillna(adad , mean,...)
data.fillna(method  ='ffill , bfill')



2- Data Type

'''


df.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   temp     4 non-null      int64
 1   modulus  4 non-null      int64
dtypes: int64(2)
memory usage: 192.0 bytes

'''

data = np.array([['10',20],['30',40] ,['40',50],['60',80]])
df = pd.DataFrame(data,columns=['temp','modulus'])


df.info()


'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   temp     4 non-null      object
 1   modulus  4 non-null      object
dtypes: object(2)
memory usage: 192.0+ bytes


object, str 



temp . modlus --> float adad bashe 
'''

#raveshe aval 
df['temp'] = df['temp'].astype(float)


df.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   temp     4 non-null      float64
 1   modulus  4 non-null      object 
dtypes: float64(1), object(1)
memory usage: 192.0+ bytes

'''




#
#df['modulus'] = df['modulus'].astype(float)

#raveshe dovom

df['modulus']= pd.to_numeric(df['modulus'])


df.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   temp     4 non-null      float64
 1   modulus  4 non-null      int64  
dtypes: float64(1), int64(1)
memory usage: 192.0 bytes

'''







'''


datato import mikoni
data = pd.read_....
data --> Pandas dataframe


4 ta msoihkele


data.info() --> Non-null (empty cell) , type


1- Empty cells --> kahlian, computeret
data.info() ---> har soton non null = len(datat)
data.isnull().sum()
data.dropna() tamame onae k kahlian miparan
data.fillna(adad , mean,...)
data.fillna(method  ='ffill , bfill')



2- Data Type
data.info() --> type ha msohekel
raveshe 1''
data['soton'] = data['soton'].astype(float)


raveshe2
data['sotoon'] = pd.to_numeric( data['sotoon'])



3- Manteghi ---> yani inke masalan temp nabayad balaye 50 bashe




4- Duplicated



'''

zarf.head()

'''
           Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0  clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1  clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2  clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3  clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4  clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04

[5 rows x 10 columns]

'''




zarf['Melting point'].describe()


'''
count    975.000000
mean     467.797056
std       70.519339
min      344.000000
25%      413.000000
50%      459.000000
75%      512.000000
max      623.000000
Name: Melting point, dtype: float64


balaye 400 -->hazf konm
-->filtering

'''

zarf['Melting point'] > 400
'''
0       True
1       True
2       True
3       True
4       True
 
970     True
971     True
972    False
973    False
974    False
Name: Melting point, Length: 975, dtype: bool

'''



zarf[zarf['Melting point'] > 400]


'''
             Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0    clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1    clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2    clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3    clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4    clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04
..            ...     ...            ...  ...   ...              ...           ...
967    Etoricoxib  358.80          407.5  ...  68.3              0.6  4.590000e-03
968    Etoricoxib  358.80          407.5  ...  68.3              0.7  1.520000e-02
969    Etoricoxib  358.80          407.5  ...  68.3              0.8  3.616000e-02
970    Etoricoxib  358.80          407.5  ...  68.3              0.9  6.137000e-02
971    Etoricoxib  358.80          407.5  ...  68.3              1.0  5.374000e-02

[792 rows x 10 columns]

'''




high_Tm_data = zarf[zarf['Melting point'] > 400]
 

high_Tm_data.head()

'''
           Drug      Mw  Melting point  ...  TPSA  EtOH mass ratio    solubility
0  clotrimazole  344.84          421.0  ...  17.8              0.0  2.920000e-08
1  clotrimazole  344.84          421.0  ...  17.8              0.1  1.180000e-07
2  clotrimazole  344.84          421.0  ...  17.8              0.2  1.260000e-06
3  clotrimazole  344.84          421.0  ...  17.8              0.3  1.290000e-05
4  clotrimazole  344.84          421.0  ...  17.8              0.4  1.120000e-04

[5 rows x 10 columns]

'''



high_Tm_data['Melting point'].describe()

'''

count    792.000000
mean     489.951806
std       58.769661
min      406.500000
25%      442.000000
50%      476.500000
75%      529.000000
max      623.000000
Name: Melting point, dtype: float64

'''




#---duplicated

df.duplicated()



df.duplicated().sum() #0
#duplicated dari



'''


datato import mikoni
data = pd.read_....
data --> Pandas dataframe


4 ta msoihkele


data.info() --> Non-null (empty cell) , type


1- Empty cells --> kahlian, computeret
data.info() ---> har soton non null = len(datat)
data.isnull().sum()
data.dropna() tamame onae k kahlian miparan
data.fillna(adad , mean,...)
data.fillna(method  ='ffill , bfill')



2- Data Type
data.info() --> type ha msohekel
raveshe 1''
data['soton'] = data['soton'].astype(float)


raveshe2
data['sotoon'] = pd.to_numeric( data['sotoon'])



3- Manteghi ---> yani inke masalan temp nabayad balaye 50 bashe
data[soton].describe()

new_data = data[data[sotone]>shart]




4- Duplicated --> radifi dashte bashdi k duplicated bashe 
data.duplicated().sum() -->age gehyr az 0 bod yani duplicateded

data.remove_duplicated(inplace=True)



yekseri az radif ha zarbe didan , hazf shodan 

data.info() -->checke nahae
data.duplicated().sum()
data.describe() --> mean , min, ....

data.reset_index(drop=True,inplace=True)

0
1
10
11
12


0
1
2
3
4


-->varede ML (machien elarning bshe)

data.to_excel()
data.to_csv()

-->edame midim niazi b in kara nis



STep 0 --> data cleaning ro baham poshte sar gozashtim 

'''


import numpy as np
import matplotlib.pyplot as plt
temp = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
density = np.array([0.55, 0.60, 0.65, 0.71, 0.76, 0.80, 0.85, 0.89, 0.92, 0.95])

density = density* 1000
plt.scatter(temp,density)
plt.title('my experimental data')
plt.xlabel('temp')
plt.ylabel('density')
plt.grid()
plt.show()



#linear --> y = a *x + b

#regresion -_> rabtee ye betne x va y [a , b]

a = np.random.uniform(0,1)
b = np.random.uniform(0,400)

print(a,b) #0.25871182063579656# 0.4330093300758725
my_y = a * temp + b 


plt.scatter(temp,my_y,label='my y')
plt.scatter(temp,density,label='Our data')
plt.title('my experimental data')
plt.xlabel('temp')
plt.ylabel('density')
plt.xlim(100,1000)
plt.ylim(200,1000)
plt.grid()
plt.legend()
plt.show()




for i in range(0,100):
    a = np.random.uniform(0,1)
    b = np.random.uniform(0,400)

    print(a,b) #0.25871182063579656# 0.4330093300758725
    my_y = a * temp + b 


    plt.scatter(temp,my_y,label='my y')
    plt.scatter(temp,density,label='Our data')
    plt.title('my experimental data')
    plt.xlabel('temp')
    plt.ylabel('density')
    plt.xlim(100,1000)
    plt.ylim(200,1000)
    plt.grid()
    plt.legend()
    plt.show()



#har 
for i in range(0,100):
    a = np.random.uniform(0,1)
    b = np.random.uniform(0,400)

    print(a,b) #0.25871182063579656# 0.4330093300758725
    my_y = a * temp + b 
    
    

    my_loss=[]
    for i in range(0,len(my_y)):
        loss=(abs(my_y[i] - density[i]))**2
        my_loss.append(loss)
        
        
    final_loss=sum(my_loss)/10000

    plt.scatter(temp,my_y,label='my y')
    plt.scatter(temp,density,label='Our data')
    plt.title(f'my experimental data : {final_loss}')
    plt.xlabel('temp')
    plt.ylabel('density')
    plt.xlim(100,1000)
    plt.ylim(200,1000)
    plt.grid()
    plt.legend()
    plt.show()
    
    
    
    
    

#hardafe
for i in range(0,100):
    # step 1 ---> a besaz , b besaz
    a = np.random.uniform(0,1)
    b = np.random.uniform(0,400)

    #strepo 2 --> ye khat ejadid besaz
    my_y = a * temp + b 
    
    
    #step 3---> loss hesab (fasele khat ba noghate vagheimo)
    my_loss=[]
    for i in range(0,len(my_y)):
        loss=(abs(my_y[i] - density[i]))**2
        my_loss.append(loss)

    final_loss=sum(my_loss)/10000

    plt.scatter(temp,my_y,label='my y')
    plt.scatter(temp,density,label='Our data')
    plt.title(f'my experimental data : {final_loss}')
    plt.xlabel('temp')
    plt.ylabel('density')
    plt.xlim(100,1000)
    plt.ylim(200,1000)
    plt.grid()
    plt.legend()
    plt.show()
    
    #step 4 --> az beyen tamame in khotot
    #ooni k final_loss az hame kamtare --> oon a, b shgo bhm bargardon
    
    
    
    
    
x=temp
y=density



#100 bar begarde , a 0 - 1 , b 0 - 4000

#ravesh haye dg ee gradient descent estefadd

#random ,a b pick 

# a , b hooshmandane taghir mdie
#fght dar soorati taghir mdie k lloss kam bshe

#--> lagorithem Gradient descent


def linear_regression(x,y):
    my_loss_dictionary=[]
    
    for j in range(0,100):
        # step 1 ---> a besaz , b besaz
        a = np.random.uniform(0,1)
        b = np.random.uniform(0,400)

        #strepo 2 --> ye khat ejadid besaz
        my_y = a * x + b 
        
        
        #step 3---> loss hesab (fasele khat ba noghate vagheimo)
        my_loss=[]
        for i in range(0,len(my_y)):
            loss=(abs(my_y[i] - y[i]))**2
            my_loss.append(loss)

        final_loss=sum(my_loss)/10000
        my_dict = {'a':a ,'b':b ,'loss' : final_loss}
        my_loss_dictionary.append(my_dict)

        plt.scatter(temp,my_y,label='my y')
        plt.scatter(temp,density,label='Our data')
        plt.title(f'my experimental data : {final_loss}')
        plt.xlabel('temp')
        plt.ylabel('density')
        plt.xlim(100,1000)
        plt.ylim(200,1000)
        plt.grid()
        plt.legend()
        plt.show()
        
        
    min_loss=10000
    best_params= {}
    for loss_key in my_loss_dictionary:
        my_loss = loss_key['loss']
        
        if my_loss < min_loss:
            min_loss=my_loss
            best_params =  {'a' : loss_key['a'] , 'b' : loss_key['b']}
    return best_params
            
        
        
        
    
    
    
linear_regression(temp,density)

#'a': 0.6424688548892594, 'b': 379.5973786409799}

#paen tarin loss ro dashte

#oon kahtie ke ma donbaleshi



#---------------

'''
pip install scikit-learn


sklearn --> ML 



'''

from sklearn.linear_model import LinearRegression

temp = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
density = np.array([0.55, 0.60, 0.65, 0.71, 0.76, 0.80, 0.85, 0.89, 0.92, 0.95])

temp = temp.reshape(-1,1)

#object misazi
model = LinearRegression()



#.fit() ---> hamin kair k ma krdimo anajm mide
#.fit(x,y)

model.fit(temp,density)

#mire koli khat ba a,b hay motefavet msikeshe
#los ro hesab mikone
#paeen tarin loss ro , a, b sh ro var mdiare
#fitiin --> fitting

# a, b mikhay??

#a
model.coef_ #array([0.45454545])
 

#b
model.intercept_ #518.0



#pishbini

#damaye 2000 bbinam chi mishe

#prediccted = 0.45454545 * 2000 + 518

model.predict(np.array(2000).reshape(-1,1)) 
#array([1427.09090909])


#yek Linear regresion fit krdim
#a , b
#pishbini krdim




from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor


model = LinearRegression()

model.fit(x,y)

#model.predict(new_x)



#deghat sehato hesab 

#chijori kar haee k krdimo dar maghale begonjim --> gozarsh konim

#jalaseye farda enshaala






























