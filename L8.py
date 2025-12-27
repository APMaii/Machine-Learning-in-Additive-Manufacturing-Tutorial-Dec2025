"""
In The Name of GOD


Created on Sat Dec 27 14:07:45 2025

@author: Ali Pilehvar Meibody




-------Review--------------

Human (en) <-----Python----> Machine (0,1 Binary)

computer ---> 
rahe 1---> python google --> isntall --> laptabemon python ro mifahme
ama koja code bznim? --> IDE (integrated development envirionement) 
--> Vs Code , Spyder , pycharm ,.... Curser [20 Euro] 



rahe 2 --> Anaconda distribution --> Anaconda
miad baraton engar chanta laoptab mirize

to laptabet -->Main (mother)
Environement misazi (Environemnt python - . -. -)
too har environement -->spyder  ..., package

anaconda --> environemnt -->create --> Python chando mikhay -->ye windows 

bad home -->environment --> spyder

spyder (main) kole 
spyder (1404)
spyder (1405) -->pythona motefavet , package ha motefavet




#----- Python --> language
man ham laptabam daratesh , ham IDE drm mitonm code bznm


Part 1 Basic python --->

1- Python built in  functions --> narenj -->tavabe az pish neveshte shdoan
--> ina ye kari --> print() input() type() len() 
https://docs.python.org/3/library/functions.html


2- Python keywords --> Banafsh --> mantegh ro beham bzni
mantegh-->python az bala b apeen az chap b rast mikhone

sharti mikhay koni ye code, ye gehsmat az code
2.1. Just if [fght baraye yeseria]
2.2. If else [dorahi]
2.3. IF elif elif elif ... else

yekseri code repapt --> Loops
2.4For loops -> static repeat , dynamic repeat ,iteration (count ,, append, orint)
2.5. While loops --> ta zamani k shart true has ....


3 - Variables --> moteghayer , zarf k chizi tosh zakhire koni
3.1. Numbers ---> int , float, complex ** + - / .... == > < 
3.2. Boolean --> True False
3.3. Str --> 'dsjdhd'  zarf[index]  zarf[index:end+1] zarf.function() lower() upper() --> khoroji midad emal nmishod

3.4. Iterables --> multiple value inside one variable
  3.4.1. List -->[] ordered(index) ,changable , allow duplicated
          zarf[index]-->elemnt zarf[index]=new_valyue , zarf.functions() -->emal mishodn  zarf.append() ,...
  3.4.2. Tuple --> () ordered, unchanagbel , allow duplicated
  3.4.3. Set --> {} unordrd , unchangabel , no duplciated
  3.4.4. Dictionary --> {'key' :'value' ,' key2'}
 
  


Part 2 Advanced Python

def --> fucntion

input ---> [] ---> output

step definition

def namefunction(vorodi1,vorodi2,...):
    body (kararo)
    
    return Khoroji


step call
namefunction(...,,...) -->ejrash mikone khrooji ro inja barmigrdone
zarf = namefunction(...,,...) 

kolan hameye python khafanish b ine k tabe benvisi
koli codo , ba y seda krdn seda bokhore


yek script .py , folder , ... --> function haro bekeshi brion bairi estefade koni

#1-----
import script
script.function()

#2------
from script import function
function()

#3-----
from script import function as f
f()

script-->khdoet dri , y4 ta code
function ahro beyenshon rado badal kjoni


organziation , company ha too jhite haye mokhtalef
koli library 

1- Pypi
2- Github
3- CHatgpt --> hite donable

-->documentation (get started, tutorial , quickstart --> yadeshon begire + GPT)
-->kahafn ha --> ketab drn 





built in libraries [vizhe roye laptobeton has]
import math 
math.sqrt(100)  --> 10
math.sin() -->sinos..

import time
import random

baghei ro nadare --> bayad va installesh koni



Part 3 - Advanced lIBRARIES
in ketabkhoen ahro laptabet ndre

toooye kernelet (shell)
pip install numpy -->bad ejaze dri az numpy estefade
pip install matplotlib
pip install pandas
pip install scikit-learn
---> PyPi
pip install pygamlab


3.1. Numpy -->number + python --> 
mesle list --> array --> faster (C++), 2d,.,..
computation.
Scipy, sympy -->ketabkhone dgan rooye numpy neveshte shodan
kar haye nuemrical modeling, differential equation

import numpy as np
#np -->fodlere numpy

zarf = numpy.array([1,2,3,4,5])

access

zarf[radif, soton]

#radif mohem tre baraye numpy

numpy.arange()
numpy.random()
numpy -->adad besaze khoshgel

reshape(radif,soton)




3.2. Matplotlib --> rasm man injam
import matplotlib
matplotlib.pyplot.plot(x,y)


import matplotlib.pyplot as plt
plt.plot(x,y)



import matplotlib.pyplot as plt

plt -->fucntion mikeshim biron
plt.function()-->brion 

khat
plt.plot(x,y,.... mfc , mec , s , ...marker, ls , )

noghat
plt.scatter(x,y, s , alpha, c)

bar
plt.bar()

plt.hist()

plt.---> m,atplotlib -->[GPT karo dar maire]


-->plt.fucntion()
optional
plt.title()
plt.xlabel()
plt.ylabel()
plt.xlim(az chan ta chan)
plt.ylim(az chan ta chan)
plt.grid()
plt.legend() --> plt.function(label)
plt.show() -->namayesh
plt.savefig()


3.3. Pandas
pandas --> man ychi mese numpy --> soton ham va radif ha -->label bandi koni

list = [10,20,30,40,]

np.array --> [10,20,30,40] sari tare , 2d 

pd.DatFrame --> [10,20,30,40] , columns=['temp' ,'speed'] , index=['']

zarf = pd.DataFrame()

#-->soton moheme

zarf[sotoon]

zarf.iloc[shoamre radif]
zarf.loc[esme radif]


zarf[soton].iloc[radife] #kodom soton kodom radif



numpy -->misaze

pd.DataFrame(zarf_numpy , columns =[])
numpy -->dataframe

numpy -->kary khdoehs, tavabe khdoesho
numpy -->mohasebate sangij\n


dataframe -->karaye khdoeshe

dartaframe --.clean krdno in chzias


pd.read_csv()
pd.read_excel()

excel,csv --> Dataframe dar biaram va badesh baahsh karamo anajm

filtering,dastresi koni taghir bdi 

.info()
.head()
.tail()
.describe()
.mean()
.statistcial --->std(),,..... --> gpt , pandas documentation 

step 0 --> ghable az Ml vaghty datto miari yeseri cleaning anjam bdi

1-Empty cell --> info() dropna() fillna()
2-Wrong format --> info() .astype() .to_numeric()
3-Manteghi --> filtering 
4-duplicated --> .duplciated().sum() .drop_duplicated()

--> 

.info()
.head()
.tail()
.columns -->khialet rahte
dataye man clean shdoe va amadas

hamin amade shdoaso

.to_excel()

.to_csv


pd.read_csv('khode fata')
pd.read_csv('clean shdoash')






"""


'''

dar donyaye mohandesi -->
masael ro bayad b x va y tabdil krd
x() -->moteghayere mostaghel (independent variable) --.daste khodemon --> Process parameter (material parameter)
y() -->moteghayede vabaste (dependent variable) --> quality parameter ->moheme , mikhaytsh analyzi ,, decision  , predict , mdoelsazi


y= f(x) --> y ye rabete ee ba x dare


sade begam

temp --> modulus 

#mano shoma nmidonim 
modulus = 20 * temp + 8


modulus = f (temp)


aluminium damahaye motefavet --> print--> modulus -->y adad



azmayeshgah ->vaght mziarim ,haznie mizarim ,

         temp      --> modulus
case 1
case 2 
sample3
sample4

try error mikni ta peydash koni -->
vaghty yedone aprameter dari , 
100 ta dari?


5 ta aprametr dari --> 10 ta range begire

10 * 10 * 10 * 10 * 10 = 100000 halato bayad test koni tabbini kodom behtrin mdoulus mide


15, 20 
50 ta nemone bznim

200



         temp      --> modulus
case 1
case 2 
sample3
sample4


'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



temp = np.linspace(100,1000,10)

modulus_list=[]

for i in range(0,len(temp)):
    random_number = np.random.uniform(10,500)
    
    
    modulus = 4 * temp[i] + 90 + random_number
    
    modulus_list.append(modulus)
    
    

modulus = np.array(modulus_list)



plt.scatter(temp,modulus)
plt.title('generated data')
plt.xlabel('temeprature')
plt.ylabel('modulus')
plt.grid()
plt.show()


print(temp)
#[ 100.  200.  300.  400.  500.  600.  700.  800.  900. 1000.]

print(modulus)
#[ 951.00505955 1346.63093748 1545.25507603 2156.43363718 2222.74847599
#2563.2850692  3363.64342708 3436.65888033 3745.20285366 4177.01140232]



#modulus = a* temp + b

#10 ta noghte
#hadaf -->man betonam y ro bar asase x dar bairam
#rabetasho-->bale rabetasho



x = np.arange(100,1000)

#y = a*x + b 
#a , b -->koliu khat bekeshi
#ba taghire a -->shiebshi\
#b --> ertefashe

y =  2 * x + 80



plt.scatter(temp,modulus,label='experiemntal')
plt.scatter(x,y,label = 'proposed line')
plt.title('generated data')
plt.xlabel('temeprature')
plt.ylabel('modulus')
plt.legend()
plt.grid()
plt.show()




x = np.arange(100,1000)

#y = a*x + b 
#a , b -->koliu khat bekeshi
#ba taghire a -->shiebshi\
#b --> ertefashe

y =  6 * x + 80



plt.scatter(temp,modulus,label='experiemntal')
plt.scatter(x,y,label = 'proposed line')
plt.title('generated data')
plt.xlabel('temeprature')
plt.ylabel('modulus')
plt.legend()
plt.grid()
plt.show()



x = np.arange(100,1000)

for i in range(1,100):
    
    y = 0.1*i * x + i*10
    
    plt.scatter(temp,modulus,label='experiemntal')
    plt.scatter(x,y,label = 'proposed line')
    plt.title('generated data')
    plt.xlabel('temeprature')
    plt.ylabel('modulus')
    plt.legend()
    plt.grid()
    plt.show()


'''

Regression-->

random --> a , b ro tenkhab mikoni


yekseri khat mikeshi --> (a1,b1)  (a2,b2) (a3,b3)

fasleye beyen oon khat ro ba noghhate --> loss function 

#loss fucntion (chan no ) -->false ye beyne noghhat ba khat , be tavabne do , radical
#be tavane do loss

#koli khat a,b haye motefavety -->loss haye motefavet

#beyneshon oni k loss e kamatri dare -->yani approximately rooye hamashon oftade --> enagr dare
#raftare inaro doros mig -->in hamoon khata k ma donabelsh bodim







'''
    
    
    

x = temp

for i in range(1,100):
    a= 0.1 * i
    b = i*10

    y = a * x + b
    
    #loss_list = y - modulus
    
    loss = (y - modulus)**2
    
    mean_loss = int(loss.mean() /100000)
    
    
    plt.scatter(temp,modulus,label='experiemntal')
    plt.scatter(x,y,label = 'proposed line')
    plt.title(f'generated data loss : {mean_loss}')
    plt.xlabel('temeprature')
    plt.ylabel('modulus')
    plt.legend()
    plt.grid()
    plt.show()
    
    


#f'

print('hi i need a apple') #hi i need a apple


a= 10

print('hi i need a apple') #hi i need a apple

    

#f' --> esme zarf

print(f'hi i need {a} apple ') #hi i need 10 apple 




#100 ta kolan khat sakht ba a , b motefavet loss ro hesba kon

x = temp

for i in range(1,100):
    a= 0.1 * i
    b = i*10

    y = a * x + b
    
    #loss_list = y - modulus
    
    loss = (y - modulus)**2
    
    mean_loss = int(loss.mean() /100000)
    
    
    plt.scatter(temp,modulus,label='experiemntal')
    plt.scatter(x,y,label = 'proposed line')
    plt.title(f'generated data loss : {mean_loss}')
    plt.xlabel('temeprature')
    plt.ylabel('modulus')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    


regression_history = []
x = temp

for i in range(1,100):
    a= 0.1 * i
    b = i*10

    y = a * x + b
    
    #loss_list = y - modulus
    
    loss = (y - modulus)**2
    
    mean_loss = loss.mean() /100
    
    
    one_dict = {'a' : a , 'b' :b , 'loss' : mean_loss}
    
    regression_history.append(one_dict)


    plt.scatter(temp,modulus,label='experiemntal')
    plt.scatter(x,y,label = 'proposed line')
    plt.title(f'generated data loss : {mean_loss}')
    plt.xlabel('temeprature')
    plt.ylabel('modulus')
    plt.legend()
    plt.grid()
    plt.show()
    
    
for dictt in regression_history:
    print(dictt['loss'])
    
    
    
min_loss = 10000000000

for dictt in regression_history:
    if min_loss > dictt['loss']:
        min_loss = dictt['loss']
        best_a=dictt['a']
        best_b = dictt['b']
        
        
        
print(best_a) #3.5
print(best_b) #350

#a, b haye motefavet 



a= 3.9000000000000004
b = 390
x= temp
y = a * x + b

#loss_list = y - modulus
plt.scatter(temp,modulus,label='experiemntal')
plt.plot(x,y,label = 'proposed line')

plt.title(f'generated data loss')
plt.xlabel('temeprature')
plt.ylabel('modulus')
plt.legend()
plt.grid()
plt.show()


#y = 3.9000000000000004 * x + 390

#modulus = 3.9 * temp + 390


'''

regression



1- y = f(x)

y = a*x + b   ag khati nabod chi?


2- y = f(x)

y = f (x1,x2,x3,x4,x5,....)



3- shoam rast migi va n behtrin khat?



-->Machine learning



'''





def fitting(x,y):
   
    regression_history = []

    for i in range(1,100):
        
        #khat haye motefavet mikeshe (a,b)
        #0.1 - 10
        a= 0.1 * i
        #10 - 1000
        b = i*10
        
        #y misaze y = a*x + b
    
        y_ = a * x + b
        
        #loss_list = y - modulus
        #faseley beyen oon khat ba noghate vaghei mano ehsb mikone ->los
        
        loss = (y_ - y)**2
        
        mean_loss = loss.mean() /100
        
        
        one_dict = {'a' : a , 'b' :b , 'loss' : mean_loss}
        
        regression_history.append(one_dict)
    
    
        plt.scatter(x,y,label='experiemntal')
        #plt.scatter(x,y_,label = 'proposed line')
        plt.plot(x,y_,label = 'proposed line')
        plt.title(f'generated data loss : {mean_loss}')
        plt.xlabel('temeprature')
        plt.ylabel('modulus')
        plt.legend()
        plt.grid()
        plt.show()
        
    #paaen tarin loss-->
    #optimization
    #minimum loss-->behtarin khat

    min_loss = 10000000000
    
    for dictt in regression_history:
        if min_loss > dictt['loss']:
            min_loss = dictt['loss']
            best_a=dictt['a']
            best_b = dictt['b']
            
            
    #behtarin khato-->
    #y = a*x + b
    
    #paramter hasho beehm pas mdie 
    #a , b
            
    return best_a , best_b
    


fitting(x=temp , y =modulus)
  
# (3.9000000000000004, 390)
    
    
def predict(new_x):
    y_pred = 3.9000000000000004* new_x + 390
    
    return y_pred
    
    
    

predict(350) #1755.0000000000002


'''

linear regression with gradient descent

a,b hads mizne yekbar 

gridsearch -->done doen search zdm tamame parametr
doros nis


intelligent tar berim jolo

a ->random
a badi

new_a = old_a + gradient Loss/A * learning_rate


-->few steps -->minimum Loss [faasleey khat ba noghate vaghe]
behtarin khat

optimization -> revolutionary (darvin) | 

a-->




'''




#LinearRegresor


#------------>
#pip install scikit-learn --> sklearn

#sklearn -->madule --> tabe , calss -> machine learning


#import sklearn
#sklearn.preprocessing.model_selection.regressor.




from sklearn.linear_model import LinearRegression

#az sklearn.madule --> class
#object = class() --> model 
#model.tabe ha

model = LinearRegression() #paarameter 

#khoroji nmide in tabe
model.fit(temp,modulus)
'''
Reshape your data either using array.reshape(-1, 1) 
if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

'''

#x --> 1 soton --> reshape(-1,1)
# x-->chan soton -->eb ndre

temp = temp.reshape(-1,1)

model.fit(temp,modulus)
#yani mire random , a, b misaze
#hey a , b ro dar rastaee taghir mide (gradient migire)
#loss ro kam kone . loss -->fassley oon khati ba a, b sakhte ba dataye to
#-->b yejae mirese behtarin khat ->behtarin a , b

#model-->hamechi inrtoo

#model (class) -->methods()  attributes 

model.coef_ #array([3.59049973])

model.intercept_ # 576.0126324600219


#modulus = 3.59049973 * temp + 576.0126324600219


#dorah pishbini konm

new_x = 350

a= model.coef_
b = model.intercept_

print(a* new_x + b) #[1832.68753664]



#---khode inakro khode mdoel --> model.predict() model.predict(new_x)


model.predict(np.array([350]).reshape(-1,1))
# array([1832.68753664])





#---------- (agha beman begoo)
#yek modire company mishine mige broo modelsazi kon

#temp 3000 dar bairim

#modlsazi oxidation sio2

#2000 -->3000


#temp          --> 500 600 700 800 900 100 ....
#thickness Sio2 -> 1   2   3    3.4 5.6  

#company mikhad --> 3000 anjam bde -->mitrse, risk kone dastgaha avaz she 

x = np.arange(800,2000,100).reshape(-1,1)

y = 0.0004 * x + 0.00000020


plt.scatter(x,y , s=30)
plt.xlabel('Temperature')
plt.ylabel('SiO2 Thickness')
plt.grid()
plt.show()



from sklearn.linear_model import LinearRegression


model = LinearRegression()

model.fit(x,y)


model.coef_ # array([[0.0004]])
model.intercept_ #array([2.e-07])

#sio2 thickness = 0.0004 * tempo + 0.0007

#--.raese company


#3000??

new_x = np.array([3000]).reshape(-1,1)

y_pred = model.predict(new_x)

print(y_pred) #[[1.2000002]]

#raeese company, regression , ..hichi nmidone
#az koja bvefahmam rast migi???

'''
3 ta damaro k nadarim (850 , 1150 , 1750) nadarim yeseri x k ndrim

bman mide --> pishbini kon --> y_pred

khodesham mire azmayeshgah test mikone --> y_true


CTO --> ekhtelafe y_true - y_pred

jofteshon menha kone --> Mean absolute error (MAE)

means square error (MSE)

mean absolute percentage error (MAPE)

R**2 


ekhtelafe beyen y_true va y_ppred

MAPE --> 4 % e ekhtelafesh --> deghatet 4% 


bejaye inke inkaro koni -->beri model besazi
bad bri pishe modir  , modiret bre bede azmayeshgah
azmayeshgah bre anma bde , cto bereanjam bde
ag bg ok 
ag bge na -->dobare az aval
dafe dovom root nmishe 


--> Train Test split



datato az hamon aval --> split kon (jodash kon)
ye ghesmatisho 20 % - 25% --> bekesh biron 
bendazsh doro -->jodsa kon

<----80%---->  <--20%-->
   Train data   Test data

 Test data -->in hamoon dataee bode k gharar bode modiret begrie
 

train data --> ino estefade mikonam modelam mibine --> miresoen cheghdr model tones khob yadbegire
test data --> new , unseen data --> geenralization predidtion anjam bde


'''


#tabe ee vojod dare --> khode sklearn -->
#x , y --> [] --> x_train, x_test , y_train , y_test

#data --> train data , test data
#x ,y --> x_train,y_train  , x_test,y_test


x = np.arange(800,2000,100).reshape(-1,1)

y_list=[]
for x_ in x:
    y = 0.0004 * x_ + np.random.uniform(0.05 , 0.1)
    y_list.append(y)

y = y_list


from sklearn.model_selection import train_test_split

#0.2 --.20%
#0.25 -->25


x_train, x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.2)




plt.scatter(x_train,y_train,label='train')
plt.scatter(x_test,y_test,label='test')
plt.xlabel('Temperature')
plt.ylabel('SiO2 Thickness')
plt.grid()
plt.legend()
plt.show()



x_train, x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.2 , shuffle = True , random_state=42)



#nokteye asli injas

from sklearn.linear_model import LinearRegression


model = LinearRegression()


#model.fit(x,y)-->kole data

#alan migam na -> x,y --> train | test --> train fght bznm 


model.fit(x_train,y_train)



#hala migi khob tamom shod 

# do ta score 


plt.scatter(x_train,y_train,label='train')
plt.xlabel('Temperature')
plt.ylabel('SiO2 Thickness')
plt.grid()
plt.legend()
plt.show()


print(x_train)
'''
[[1600]
 [1300]
 [1000]
 [ 900]
 [1900]
 [1200]
 [1500]
 [1100]
 [1400]]

'''



y_train_pred = model.predict(x_train)


plt.scatter(x_train  , y_train , label = 'true')
plt.plot(x_train ,y_train_pred , label='predicted' )

plt.legend()
plt.grid()
plt.show()



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_percentage_error


#funct(true , predic) -->hesab


mean_absolute_error(y_train , y_train_pred) # + - 0.009379386099546867

mean_absolute_percentage_error(y_train, y_train_pred)# 0.01 --> 1/100 --?> 1%


# train_score --> data hae k dide khati k sakhte inghad fasele --> pishbiniu --> cheghad amozesh dide


#train score --> train data + khate --> modelemon chghd fahmide va amoozesh dide (1% , 40%)


plt.scatter(x_test,y_test)
plt.show()



y_test_pred = model.predict(x_test)


plt.scatter(x_test,y_test)
plt.plot(x_test , y_test_pred)
plt.show()



mean_absolute_error(y_test ,y_test_pred ) # 0.010421908087277332

mean_absolute_percentage_error(y_test, y_test_pred) #0.022175953432972323

#test score


'''
x,y --> x_tst, y_test |||| x_train , y_train


Train Score --> y_train  y_predic -->cheghad modele man amzoesh dide 


Test Score ---> y_test y_pred --> prediction ro mifhme



train score bala nabashe test score nmitone bala bshe --> underfitting --> fitt nashdoe yad ngrfte

ag train score kh kh kh kh bala bshe --> test score bai paeen --> overfittin --> hata noise .. nabayd yad bgire gomrahe bias yad grft

trade off --> train score , test score

'''



#========== FULL PIPELINE
#---import ------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#======= STEP 0 ------
#biaramehs dakhel
data = pd.read_excel('/Users/apm/Desktop/DOE.xlsx')



data.head()

'''
   No.  POWER  HATCH  Speed  L.Th    VED  Relative Density      BR
0    1    160  0.115   1200  0.03  38.65         99.160000  14.904
1    2    160  0.105   1200  0.03  42.33         98.906528  13.608
2    3    180  0.115   1200  0.03  43.45         99.068873  14.904
3    4    160  0.100   1200  0.03  44.44         98.910660  12.960
4    5    160  0.115   1000  0.03  46.38         98.778294  12.420

'''



data.info()

'''
276 ta data , 8 ta soton , non-null, typa 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 27 entries, 0 to 26
Data columns (total 8 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   No.               27 non-null     int64  
 1   POWER             27 non-null     int64  
 2   HATCH             27 non-null     float64
 3   Speed             27 non-null     int64  
 4   L.Th              27 non-null     float64
 5   VED               27 non-null     float64
 6   Relative Density  27 non-null     float64
 7   BR                27 non-null     float64
dtypes: float64(5), int64(3)
memory usage: 1.8 KB

'''


data.describe()

data['POWER'].describe()
'''
count     27.000000
mean     181.111111
std       15.525001
min      160.000000
25%      170.000000
50%      180.000000
75%      200.000000
max      200.000000
Name: POWER, dtype: float64

'''

data.columns
'''
Index(['No.', 'POWER', 'HATCH', 'Speed', 'L.Th', 'VED', 'Relative Density',
       'BR'],
      dtype='object')

'''


data.drop(columns='No.',inplace=True)


#data.drop(columns='L.Th',inplace=True)

#data.drop(index=8)

#data.reset_index(drop=True,inplace=True)



data.to_excel('cleaned_DOE.xlsx')




data.plot(kind='scatter',x='POWER' , y='VED' )
plt.show()


plt.scatter(data['POWER'],data['VED'])
plt.show()



data.columns
#========STEP1 ---> x , y -->numpy array -->computation

x_data = data.drop(columns=['VED','Relative Density','BR'])
y_data = data['VED']



x = np.array(x_data) #.reshape(-1,1)
y = np.array(y_data)





from sklearn.model_selection import train_test_split


x_train, x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2 ,shuffle= True , random_state=42)





from sklearn.linear_model import LinearRegression


model = LinearRegression()

#----training----
model.fit(x_train,y_train)

#a , b
model.coef_ #array([ 3.13907306e-01, -6.10436196e+02, -6.21688591e-02, -3.97007260e-28])

#y = a* power + b* hatch + 
model.intercept_ # +

y_train_pred = model.predict(x_train)


from sklearn.metrics import mean_absolute_percentage_error


train_score = mean_absolute_percentage_error(y_train,
                                             y_train_pred )


print('train score : ', train_score) 
#train score :  0.022369689322501503



y_test_pred = model.predict(x_test)

test_score = mean_absolute_percentage_error(y_test,y_test_pred)


print('test scoore : ',test_score)

#test scoore :  0.04395393623851188


#==================================================
#==================================================
#==================================================
#==================================================
#==================================================
#==================================================
#==================================================
'''

ProtoType


DATA ----> pd.read_excel() read_csv() -->vared mikonim

data cleaning (L7) --> 



#soton ha --> sotoni k daste hsoams taghir mitonid bdid
#process parameter , material proeprty ,...--> X (vorodi ha)

#shoam moheme, analyz , predict , --> quality parameter 


x --> (np.array())
y --> np.array()



#---train_test_split function from sklearn

test | train


#--->model az sklearn miarid biron
model = class()


model.fit(train ha)

pred = y_train --> model.predict()


MAE , MAPE -->score = y_train , y_pred_train --> amoozesh dide


model.predict() --> test ha

score = y_test , y_pred_test --> darsad pishbinio neshon mide


'''
#----------------------------------
#------------Imports -----------------
#----------------------------------



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('/Users/apm/Desktop/DOE.xlsx')
#----------------------------------
#------------CLeaning -----------------
#----------------------------------
data.head()
data.tail()
data.info()
#--> data.drop() , data.dropna() data.fillna()
#data.astype() .to_numeric()
data.columns
#ag ok bood 

#--------------------------------------------
#------------Data --> np array x , y -----------------
#--------------------------------------------
x = np.array(data[['POWER', 'HATCH', 'Speed', 'L.Th','VED','BR']])

y = np.array(data['Relative Density'])

#----------------------------------
#------------Train test split -----------------
#----------------------------------

from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , shuffle=True, random_state = 42)



#----------------------------------
#------------Model selcctiopn -----------------
#----------------------------------
from sklearn.linear_model import LinearRegression


model = LinearRegression()


'''
marooof darim

#esme model
1-Linear
2- K neearst neighbour
3- Decision tree
4- Random forest
5- support vector machine
6- multi layer perceptron (artificial neural network)


#----regressor --> y ha estefade continious
linearregresion
KNNregressor
decisiontreeregressor
randomforestregressor
svr (supporrt vector regressor)
MLPregressor



#classifier
Logisticregression
KNNclassifier
decisiontreeclassifirer
randomforestclassifier
svc (support vector classifier)
MLPclassifier




Linear regression--> 
a, b --> khat haye motefavet o mizad --> gradien t descent -->los hesba 
rabetey beyne x va y ro befahme
codesho nvshtm , bazam sakhte bazam kasi 90% 



riazish mishe  -->model riazish
ag bdoni chikar mikone -->farghi nmikone
az model ha --> har 6 taro entekhab --.comapare
train score test score
farghi --> too deleshon parsmter -->hyperparametr --.fara parameter
tasir mizare rooye yadgirish


decision tree max depth = 2 --> test score 0.3 | 8 test score = 0.1





'''



'''
Model haye motefaveti darim



Machine learning b 3 ghesmat taghsim mishe


#----> x, y 

1- Supervised learning ( tahte nezarat)
ham x darim ham y ro darim
x__.temp ,.... y-->
rabete ye beyne x va y hastim


xxxx       y 

y = f (x)   
Nesbat ink y chie

--> 1.1. Y continious -->1.4334 --> ML Supervisied regression  --> model regressor


    1.2. Y categorial --> 0 1 , bale kheyr, awli bad khob, keyhole ..  --> classification 9daste bandi
    -->supervised classification --> modl -->classifier





2- Unsupervsied learning (gheyre tahte nezar)
XXXX y XXXX  --> unlabeled

yeseri x hastan --> daste bandishon 
clustering konim

K mean 
DBSCAN 
gussian mixture 

koli ma defect -> size haye motefaveti 
Tomography 1000 defect hast -->         y
1  x y z sphericity gap volume ....  xxxKeyhole  xxxx
2                                     
3
4
5
...
1000





#1,2 -->dataro dashtim midadim b model
#model -->train --> pishbini


# data besorate stream maid b reinforcmeent
#--> data maid -->hey bishtatr yad

#agent , environement , padash 
#padash , mojazat 
#robotic o ....
3- Reinforcement learning
#bahse beshedat pichide o riazi 



'''




#----------------------------------
#------------Imports -----------------
#----------------------------------













#-------------------
'''
PROTOTYPE
'''
#--------------------


#----------------------------------
#------------Imports -----------------
#----------------------------------
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('/Users/apm/Desktop/DOE.xlsx')
#----------------------------------
#------------CLeaning -----------------
#----------------------------------
data.head()
data.tail()
data.info()
#--> data.drop() , data.dropna() data.fillna()
#data.astype() .to_numeric()
data.columns
#ag ok bood 

#--------------------------------------------
#------------Data --> np array x , y -----------------
#--------------------------------------------
x = np.array(data[['POWER', 'HATCH', 'Speed', 'L.Th','VED','BR']])

y = np.array(data['Relative Density'])

#----------------------------------
#------------Train test split -----------------
#----------------------------------

from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , shuffle=True, random_state = 42)



#----------------------------------
#------------Model selcctiopn -----------------
#----------------------------------

#Y --> continous 
#========REGRESSOR ====================

from sklearn.linear_model import LinearRegression
model = LinearRegression()


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=10)
model = KNeighborsRegressor(n_neighbors=3)


from sklearn.tree import DecisionTreeRegressor
#ta che omghi derakht besazd
#harhvci bsihtar bashe compelx

model =DecisionTreeRegressor(max_depth=3)



from sklearn.ensemble import RandomForestRegressor
#yek jangali az decision tree n_estimator

model = RandomForestRegressor()


from sklearn.svm import SVR
# c , gamma m kernal
model = SVR()


from sklearn.neural_network import MLPRegressor
#hidden layer size
model = MLPRegressor()


#=====Y --> labeli , goroh 
#-------classificatuion--------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


from sklearn.tree import DecisionTreeClassifier
model =DecisionTreeClassifier()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


from sklearn.svm import SVC
model = SVC()


from sklearn.neural_network import MLPClassifier
model = MLPClassifier()





#----------------------------------
#------------Fitting -----------------
#----------------------------------

#training datsaet -->fit konim

model.fit(x_train,y_train)


#lr --> a, b mizado y raveshi
#knn --> BLACK BOX --> az tarigeh ravesh khdoeshon 
#a , b nadare , parametr haye khdoesh --> baklck box 
#model.predict()


#---------------------------------
y_pred_train= model.predict(x_train)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

train_score_mae = mean_absolute_error(y_train ,y_pred_train )
train_score_mape = mean_absolute_percentage_error(y_train ,y_pred_train )

print('train score MAE:',train_score_mae)
print('train score MAPE:',train_score_mape)
'''
train score MAE: 0.16745643790406528
train score MAPE: 0.0016923729951215378 0.1 %


kh khoob amozsh dide --> kh inam khoib bashe -->bade




KNN
train score MAE: 0.17839553685578902
train score MAPE: 0.0018019126198267022


KNN N =10
train score MAE: 0.1760474716955468
train score MAPE: 0.0017784661407663261

KNN N =3 
train score MAE: 0.1668185650293601
train score MAPE: 0.001685025469524079





'''


#-------- TEST --> ACCURACY 

y_pred_test = model.predict(x_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

test_score_mae = mean_absolute_error(y_test ,y_pred_test )
test_score_mape = mean_absolute_percentage_error(y_test ,y_pred_test )

print('Test Score MAE:',test_score_mae)

print('Test Score MAPE:',test_score_mape)

'''
Test Score MAE: 0.29682774198497697
Test Score MAPE: 0.0029917502639913323

pishbinito neshon mide



KNN
Test Score MAE: 0.23623659253469498
Test Score MAPE: 0.0023816449426614194



DT
Test Score MAE: 0.3553474676716443
Test Score MAPE: 0.0035840476296038026



'''

#ok shod
#azash estefade koni

#range

new_x
y_prd=model.predict(new_x)


'''
cllassifciation

y_pred -->adad nis - oonyeki adad /100


y _pred =  0 0  0 0 0 1 1 1 1 0 0 0 
y_true = 1 00 0 0 0 0 0  0 0 

accuracy

chan darsad doros gofte

'''
#from sklearn.metrics import accuracy_score
#mean_absolutw
#accuracy_score()





#model.fit()
#model.predict()

#
'''

Whatsapp --> file --> 
gpt --> kamel
hyperparamtr haye in 6 model kodoma mohemtrinan 
va range ha chie



--> file amade -->github --> mohemtrin haro 

ghablesh khdodeton ino bznim



'''




#========================
#========================
#========================

'''
aval ghablairo emtehan konid , test konid bad biayd inja




ADVANCED  Prototype


'''

#----------------------------------
#------------Imports -----------------
#----------------------------------
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel('/Users/apm/Desktop/DOE.xlsx')
#----------------------------------
#------------CLeaning -----------------
#----------------------------------
data.head()
data.tail()
data.info()
#--> data.drop() , data.dropna() data.fillna()
#data.astype() .to_numeric()
data.columns
#ag ok bood 

#--------------------------------------------
#------------Data --> np array x , y -----------------
#--------------------------------------------
x = np.array(data[['POWER', 'HATCH', 'Speed', 'L.Th','VED','BR']])

y = np.array(data['Relative Density'])


#----------------------------------
#------------Model selcctiopn -----------------
#----------------------------------

from sklearn.linear_model import LinearRegression
model = LinearRegression()


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()


from sklearn.tree import DecisionTreeRegressor
#ta che omghi derakht besazd
#harhvci bsihtar bashe compelx
model =DecisionTreeRegressor()


from sklearn.ensemble import RandomForestRegressor
#yek jangali az decision tree n_estimator

model = RandomForestRegressor()


from sklearn.svm import SVR
# c , gamma m kernal
model = SVR()


from sklearn.neural_network import MLPRegressor
#hidden layer size
model = MLPRegressor()


#=====Y --> labeli , goroh 
#-------classificatuion--------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


from sklearn.tree import DecisionTreeClassifier
model =DecisionTreeClassifier()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


from sklearn.svm import SVC
model = SVC()


from sklearn.neural_network import MLPClassifier
model = MLPClassifier()



from sklearn.model_selection import GridSearchCV

param_grid={
    'n_neighbors':[1,2,3,4,5,6,7,8,9,10]
    }


gs = GridSearchCV(model,param_grid , scoring='neg_mean_absolute_percentage_error' ,cv=5)


'''
kazeb nis -->
ghaboie kazeb bood--> fght 20% data
statistically 

test_size

cross fold 

cv = 5 ghesmat

5 ta alptob

10 ta 
'''
#MAPE
#neg_mean_absolute_percentage_error

#MAE
#neg_mean_absolute_error



'''
model -->on bala entkehab krdi

param_grid -->dictiomnary hast az range 




'''
'''
miad dore mdoelet bejay emodel

azinja

'''


gs.fit(x,y)


#--->

gs.best_params_ #{'n_neighbors': 8}
gs.best_score_ # -0.001965560890015618


rsults= gs.cv_results_


pred_y = gs.predict()

    
'''
raveshe 1 --> model , test train --> train score, test score


raveshe 2 --> test train --> fold taghsim mikone, chanbar train mibine 
     ejaze mdie rangi az parametr haye mdoel ro hey train koni va behtrin ro peyda koni




Prototype_ML.py





prototype_GS.py



result --> jadval moarefi
ravehs haye motefaveti vojod baraye
moarefie model , rsult ha , .....

--> Journal ha , article




'''
    
    
