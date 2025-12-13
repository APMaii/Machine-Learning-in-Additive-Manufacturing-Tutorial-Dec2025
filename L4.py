"""

In The Name of GOD


Created on Sat Dec 13 15:58:56 2025

@author: Ali Pilehvar Meibody



L4 -----





"""



'''

Overview---------------------------


Python language --> vocabs, grammar

1- Python built in functions
print() input() len() type() open() abs()
nareni , 100 ta tabe

https://docs.python.org/3/library/functions.html



2-Kyewords --> Banafsh --> Logic python ro avaz koni
logic --> python az bala b paeen , az chap b rast mesle yek ensan
coide ro khat b khat mikhone run mikone, b error bokhore stop mikone

in raveshi, logic -> beham bezani

yek khat, ya yekseri khat code ro shartish koni (dependent)
    2.1.conditional statement 
    2.1.1. Just if
    2.1.2. If else
    2.1.3. If elif else

     
    2.2. LOOps --> static repeat, dynamic repeat , iteration
    2.2.1. For loops
    2.2.2. While loops



banafsh, narenji --> sefid --> unreserved words -->python nemishnase

---> be onvane ysme yek zarf dar nazar migire
in zarf esm = meghdar dare
3-variables (moteghayer) [zarf]
3.1. Numbers (int, float, complex) 
a=10 zarfi bename a msiazim 10

c = a + b 
** * / + -   % (baghimondaro)

20%2 --> 0 

== != > < -->  True False moghayese 

3.2. Boolean (True, False)

3.3. Str -> reshte haee az character ha hastand

3.3.1 Zarf[index]  zarf[start:end+1] zarf[start:end+1:step]
3.3.2. Str functions --> zarf.function()
--> emal nmishod khoroji midad

new_str = zarf.lower()

zarf.upper() zarf.title() zarf.capitalizie()

new_zarf = zarf.replace('old' ,'new')

zarf.count('a') -->
zarf.find('a')

zarf.islower() --> True False


3.4. Iterables --> chanta value dakhele yek zarf brizi
3.4.1. List --> [10,20,30,'vahid', 10223]
      ACCESS zarf[index]
      slicing zarf[start:end+a]
      changable zarf[index] = 'new_element'
      list functions --> zarf.function() -->emali bod , khoroji nmidad
      zarf.append('element') tahe on zarf
      zarf.insert(index,element)
      zarf.remove('element')
      zarf.pop('index')
      zarf.sort() --> sort mikard
      zarf.reverse() -->reverse mikard
      zarf.clear() --> zarf = []
      del zarf  -> XXX zarf is not defined
      
3.4.2. Tuple --> (10,20,30,) ->
ordered(index) , unchangable , allow duplicated
Database o inchiza estefade msihe k data dar transfer nabayad
change beshe

3.4.3. Set --> {10,20,30}-->
unordered , unchangable, no duplicated
jahaee k aksaran --> tekrari nabayad 

3.4.4. Dictionary 

zarf = {'key':'value' , 'key2':'value2'}

sample2 = {'temp':100 , 'speed':'200'}

sample2['temp'] -->100
sample2['speed']-->200

information ha 

index value
...   ..

key value

zarf['name']
zarf['ghad']
zarf['mahsool']
zarf['dama']



'''


#--------------------
#----2 . Keywords


#----Conditional statament --> yek khat code ro sharti koni




print('salam')


name = input('salam esmet chie? :')

sen = input('senet cheghadre?:')
sen = int(input('senet cheghadre?:'))




#sen = 20
#....



sen = 10
if sen>18:
    print('salam')




sen = int(input('senet cheghadre?:'))
if sen>18:
    print('salam')






#------

#hamishe harchi sen bzni --> salam print mishe

#man mikham shartish konm 

#sharti koni , yek kaht codo ya chan khat --> if 


sen = int(input('senet chegahdre?:'))

#----1-just if 
#yshabihe ye rah zane

if sen>10:
   print('salam')

#chanta khat code inja bzarid
#shart True False
#and , or -> chnata shart bzari



'''

yek dastgahe EBM darim tarahi mikonim 


------------------
|  speed :       |
|  temp :        |
|  Material:     |
|                |
|                |
|       RUN      |
|----------------|


GUI --> graphical user interface 
rabete graphicie karbar

CLI command line interface
terminal




CLI 






etelaat ro grftim az user 


1-strategy ------------
check konim age goft yes --> run konim karo
ag nagof na 



2-strategy ------------
ag trf nevesht yes --> run bshe
ag trf nvsht no --> ok I dont start


3-strategy sevom ----->
hey az trf beporse hey bge run konm
ta zamni k trf rooye yes bzne
hey dobare beporse


'''


print('================')
print('Welcome to AmPro lab')
print('-------------')

material = input('insert the name of your material:')
process_type = input('enter your process(LPBF or EBM):')
speed = input('insert speed of your process:')
temp = input('enter your temperature:')
run = input('are you ready for running? (yes/no)')




#========== Strategy 1----------------
'''

Age --> if ......


1-just if 2-if else 3-if elif ???

rahzan
dorahi
dorahi haye too dar too?


'''



print('================')
print('Welcome to AmPro lab')
print('-------------')

material = input('insert the name of your material:')
process_type = input('enter your process(LPBF or EBM):')
speed = input('insert speed of your process:')
temp = input('enter your temperature:')
run = input('are you ready for running? (yes/no)')




if run == 'yes':
    print('starting process ......')






#----strategy 2 -------
#ag yes --> bege start
#ag no --> start 

print('================')
print('Welcome to AmPro lab')
print('-------------')

material = input('insert the name of your material:')
process_type = input('enter your process(LPBF or EBM):')
speed = input('insert speed of your process:')
temp = input('enter your temperature:')
run = input('are you ready for running? (yes/no):')


if run == 'yes':
    print('starting process ....')
    
else:
    print('Completed ... Finished')


#Completed ... Finished


#-----

#ta zamani k trf naneveshte yes hey bpors hey bepors

#ta zamani ?? 

#--agar--- --> if ha
#--repeat , boro iteration ,begard , done doen check koen -- -> for
#ta zamani k --> while


'''
ta zamani k shart == true bashe
kare zir ro anjam mide

while shart:
    yek kare
    
    


'''



while True:
    run = input('are you ready for running? (yes/no):')
    if run=='yes':
        break
    else:
        continue
    
    
print('starting process')






#---------------------------------
#---------------------------------
#---------------------------------
#---------------------------------

print('================')
print('Welcome to AmPro lab')
print('-------------')

material = input('insert the name of your material:')
process_type = input('enter your process(LPBF or EBM):')
speed = input('insert speed of your process:')
temp = input('enter your temperature:')
run = input('are you ready for running? (yes/no):')

#YES Yes yeS YES yes    yes --> 'yes'

if run.lower().strip() == 'yes':
    
    #check konim dama balaye 2000 nabashe 
    #ga bood ekhtar bdim
    
    if int(temp)>2000:
        print('Warning : your temperature exceeded 2000 Centigrade')
    
    else:
        print('starting process ....')
    
    
    
else:
    print('Completed ... Finished')



'''
answer='yes '
new_answer=answer.strip()
print(len(answer)) #y e s space 4
print(len(new_answer)) #3
'''

#2000 --> warning
#dama 1000 - 2000 ok
#1000 --> kame
'''
age haye too dar too mziani

chekc koni balaye 2000 --> warning
age paeen tre--> aya hala balaye 1000?

if elif


'''





print('================')
print('Welcome to AmPro lab')
print('-------------')

material = input('insert the name of your material:')
process_type = input('enter your process(LPBF or EBM):')
speed = input('insert speed of your process:')
temp = input('enter your temperature:')
run = input('are you ready for running? (yes/no):')

#YES Yes yeS YES yes    yes --> 'yes'

if run.lower().strip() == 'yes':
    
    #check konim dama balaye 2000 nabashe 
    #ga bood ekhtar bdim
    
    if int(temp)>2000:
        print('Warning')
        
    elif int(temp)>1000:
        print('Ok')
    else:
        print('So cold . change')
        

else:
    print('Completed ... Finished')




#-------------------------------------
#for -->

for i in [1,2,3,4,5]:
    print('salam')
    
'''

i = 1 --> print(salam) -->salam
i=2 -->print(salam) -->salam



-->
slam
salam salam salam salam

'''

#static repeat
for i in range(0,101,2):
    print('salam')


#dyanmic rpeat
for i in range(0,101,2):
    print(i)
    
    
'''
0
2
4
6
8
10
....
80
82
84
86
88
90
92
94
96
98
100
'''
    
    
#repeat --> iteration varesi

#for i in []:
    
    
#listi dari

#list mtione az adad
#az asami
#az sample ha
#density ha mitone bashe

#List --> liste smaple hat

#for zadan beri tooyhe done donasho 

#checkeshon koni

asami = ['ali','vahid','hamid','abbas','abas' ,'fateme']


#done done tooye in asami check konam
#andazashono oprnt konm 
#repeat

#for --> done done
for i in [1,2,3,4,5]:
    pass
    
 
mylist=[1,2,3,4,5]
for i in mylist:
    pass
    
    
mylist = ['ali','vahid','hamid','abbas','abas' ,'fateme']
for i in mylist:
    pass

    
asami = ['ali','vahid','hamid','abbas','abas' ,'fateme']
for i in asami:
    pass
    



asami = ['ali','vahid','hamid','abbas','abas' ,'fateme']
for i in asami:
    print(i)

'''
ali
vahid
hamid
abbas
abas
fateme

'''

asami = ['ali','vahid','hamid','abbas','abas' ,'fateme']
for j in asami:
    print(j)



asami = ['ali','vahid','hamid','abbas','abas' ,'fateme']
for esm in asami:
    print(esm)

'''
esm = ali , print(esm)  -->print(ali)


'''





asami = ['ali','vahid','hamid','abbas','abas' ,'fateme']
for i in asami:
    name_len=len(i)
    print(name_len)

'''
i=ali -->  3
i= vahid --> 5


3
5
5
5
4
6


'''


#---englishi

name_list = ['ali','vahid','hamid','abbas','abas' ,'fateme']

for name in name_list:

    print(name)

'''
ali
vahid
hamid
abbas
abas
fateme

'''

name_list = ['ali','vahid','hamid','abbas','abas' ,'fateme']
#shart True False

#len(name)>4 -->True False


for name in name_list:
    if len(name)>4:
        print(name)


'''
name = ali --> if len(ali)>4: print(ali) --> ....

'''


'''
vahid
hamid
abbas
fateme

'''

#bafraye yeseriaa-->yeseria

#





#boro fght esm haee k baltar az 4 harf daran ro print kon
'''

tooye ye loist begard ,

for , if ha estefade ---> iteration



for 

'''










'''
done doen elemnt haye k to in list









#tooye for --> if , .....


densities = [90 , 95 , 98 ,80 , ]



'''


densities = [90 , 95 , 98 ,80 ,100 , 84 , 3 , 10 , 40 , 60 ]


#boro fght oonae k balaye 80 % hastan ro print kon
#toosh --> iteration

for i in densities:
    pass

#zarfi baenam ezsampel
#yebar 90 , 95, 98 ,....

for sample in densities:
    print(sample)
    
'''
90
95
98
80
100
84
3
10
40
60

'''


for sample in densities:
    if sample>80:
        print(sample)
    
'''
90
95
98
100
84

'''


print('ali','salam','khodafez')

#ali salam khodafez

print('ali salam khodafez')


print('ali',material,'khodafez')



temp_list = [2010 , 400 , 3000, 1010 , 250 , 1500 , 2800]

'''
temp = 2010 --> erjra
temp -_>40 -->jra

'''


temp_list = [2010 , 400 , 3000, 1010 , 250 , 1500 , 2800]


for temp in temp_list:
    if temp>2000:
        print('Warning:',temp,'is very high')
    
    elif temp>1000:
        print(temp,'is ok!!')
    
    else:
        print(temp,'is very cold for process')

'''
Warning: 2010 is very high
400 is very cold for process
Warning: 3000 is very high
1010 is ok!!
250 is very cold for process
1500 is ok!!
Warning: 2800 is very high

'''



forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
distance = [20 , 30 , 40 ,50 ,70 , 80 ,20 , 30]

#force / dispalcement --> stres --> liist 



#i --> 0 ,...7 

for i in range(0,len(forces)):
    stress = forces[i] / distance[i]
    print(stress)
    
    
'''

i->0 , stress= forces[0]/ sdistance[0] = 100 / 200 = 5 -->print(stress) print(5) -->5

'''
    


#print balad bodim

#shoma mikhay --> jodash koni brizi y liste dg




forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
distance = [20 , 30 , 40 ,50 ,70 , 80 ,20 , 30]

stress_list=[]


for i in range(0,len(forces)):
    stress = forces[i] / distance[i]
    stress_list.append(stress)
    



print(stress_list)
#[5.0, 6.666666666666667, 7.5, 1.2, 1.0, 5.0, 22.5, 20.0]


#-----------------------

#----------------
#----Iteration mizanim

#for zarf in list 

#zarf = element haye list

#code ejra konim

#if , elif , print ,.....

#-----sade tarin halat

forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]

for force in forces:
    print(force)

'''
100
200
300
60
70
400
450
600

'''
forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]

for force in forces:
    new_force = force/10
    print(new_force)

'''
10.0
20.0
30.0
6.0
7.0
40.0
45.0
60.0

'''


#if else .......

#------print ......

forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]

for force in forces:
    if force > 90:
        print(force)

'''
100
200
300
400
450
600

'''
#beshmor chantas ....

forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
count =0

for force in forces:
    if force > 90:
        #print(force)
        count = count + 1
        
        
'''
zarfi bename force --> [100, 200 , 300, 60 , 70 , 400 , 450 , 600]

force = 100 -->code ejra --> True --> count = count +1 = 0 +1 = 1
force = 200 --> True --> count = count +1 = 1 + 1 = 2 
force = 300 --> True --> count = count + 1 = 2 +1 = 3

force = 60 --> jera -> force>90 , 60?90 False --> ejra  hmon 
force = 70 --> ejra force>90 --> 70>90 --> false --. hamon

force = 400 --> force >90 400>90 True ->count = count +1 = 3 +1 = 4





'''
        

print(count) #6


#sevomin halat --> bareshond ri brizi toye list

forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]

good_forces=[]


for force in forces:
    if force > 90:
        #print(force)
        #count = count + 1
        good_forces.append(force)
        
        
print(good_forces) #[100, 200, 300, 400, 450, 600]

#------ hazf kon

for force in forces:
    if force > 90:
        #print(force)
        #count = count + 1
        forces.remove(force)
        
        
        

        
        

#---------------

forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]

for force in forces:
    if force > 90:
        print(force)

#force = elemnt, 100 , 200 , 3000



#i --> 0 , 1 , 2 , ..len-->8,9

for i in range(0,len(forces)):
    if forces[i] > 90:
        print(forces[i])


#-----



forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
distances = [20 , 30 , 40 ,50 ,70 , 80 ,20 , 30]



for i in range(0,8):
    pass
    
    
for i in [0,1,2,3,4,5,6,7]:
    pass



forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
distances = [20 , 30 , 40 ,50 ,70 , 80 ,20 , 30]
allowable_stress = 7
good_stress_list=[]
for i in range(0,len(forces)):
    stress = forces[i] / distances[i]
    if stress < allowable_stress:
        good_stress_list.append(stress)
        
    

print(good_stress_list)    
#[5.0, 6.666666666666667, 1.2, 1.0, 5.0]


'''
1-Psudocode --> az bala ta paeen 1800 khat minevsuim

injori code nmiznaan

2- Function based



3- OOP (object orinted programming)


'''


'''

capsulation -->
1- code tamzi neveshte bshe
2- debugging rahat tar 
3- dynamic flexible 
4- repat


--> mazayaye function

BOXi az code --> badane
vorodi ---> Body ----> Khoroji


Input --> BOX ---> Output


function --> 
1-name (esmeshe)
2-input --> arguments
3-code dakheli --> body
4-khoroji --> output


#name zarf --> functin


def name(vorodi1)


def name(vorodi1,voirdoi2 , vorodi3):
    body
    body 
    body
    

'''

#f = m * a 



#newton 

'''



m and a --> BOX --> f mide


step1 --> defintion
step2 --> call (use)


'''

#euler #NameError: name 'euler' is not defined

#definition
def newton(m,a):
    f = m * a
    print(f)
    
    
#newton()
    
#toye python --> newton esmi vojod
    
'''




m , a --> BOX[f = m*a] --> f




'''

#euler(10,2) #NameError: name 'euler' is not defined
newton(10) #TypeError: newton() missing 1 required positional argument: 'a'
newton(10,2,19,29) #TypeError: newton() takes 2 positional arguments but 4 were given

newton(10,2) #20

#m=10 , a=2


 
newton(40,50) #2000


newton(1000,20) #20000





#------- Khoroji bvoropdi

#khoroji ba print kh kh frgh mikone

#print --> khoroji nis

#print -->namayesh
#khoroji yani --> jae k sedash kardi chizi pas bde

def newton(m,a):
    f = m * a
    print(f)
    
newton(10,2) 
    
    
zarf = newton(10,2) 
    

print(zarf) #None





def newton(m,a):
    f = m * a
    return f
#harja sedat zad ino bezar jash



zarf = newton(10,2)
    
print(zarf) #20
    
    
    
def newton(m,a):
    f = m * a
    print(f)
    return f



zarf= newton(10,2)


#zarf misaze 
#tabeye newton --> vojod dare? bale

#chnata vorodi migirer 2 ta mige ok

#2 taro miad
#m=10 , a=2 

#f = m*a = 10*2 =20 print(f) 20

#return f --> return 20 
#koja jae k seda jorde
#yani
#zarf = newton(10,2)
#zarf = 20
#toye zarrf 20 ro mirize
'''

def name():
    body
    
    
vorodi , khoroji ndshte bashe mohem nist
body dashte bashe


4 no hastan

'''


#1-na vorodi na khoroji

def welcome():
    print('salam khosh omadid')



welcome()

zarf = welcome()
print(zarf) #None


#2- vorodi dare , khoroji nadare

def newton(m,a):
    f= m*a
    print(f)
    
#vorodi dare
newton(10,2) #20

zarf = newton(10,2)

print(zarf) #None

#3- ham vorodi ham khoorji

def newton(m,a):
    f = m*a
    print(f)
    return f
    
zarf = newton(10,2)


#4-vorodi nadare, khoroji dare

def pi():
    return 3.14


number = pi()


pi = 3.14





'''


Input ---> BOX ---->Output


def name(arg1,arg2,arg3,....):
    body
    body
    print.....
    return ...



'''


def newton(m,a):
    f = m*a
    return f
    c= 100
    print(c)
    
    
    
newton(10,5)
#c , print(c) -->ejra nmsihe




forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
distances = [20 , 30 , 40 ,50 ,70 , 80 ,20 , 30]

allowable_stress = 7

good_stress_list=[]

for i in range(0,len(forces)):
    stress = forces[i] / distances[i]
    if stress < allowable_stress:
        good_stress_list.append(stress)
        
print(good_stress_list)




#tabe bood -->
'''

allowable stress -->[EJRA MIKARD ]-->good_stress_list

'''


def calculation_stresses(allowable_stress):
    forces = [100, 200 , 300, 60 , 70 , 400 , 450 , 600]
    distances = [20 , 30 , 40 ,50 ,70 , 80 ,20 , 30]
    good_stress_list=[]

    for i in range(0,len(forces)):
        stress = forces[i] / distances[i]
        if stress < allowable_stress:
            good_stress_list.append(stress)
            
    #print(good_stress_list)
    return good_stress_list




zarf = calculation_stresses(7)
zarf = calculation_stresses(allowable_stress=7)

print(zarf) #[5.0, 6.666666666666667, 1.2, 1.0, 5.0]



zarf = calculation_stresses(5)
print(zarf) #[1.2, 1.0]

zarf = calculation_stresses(1.1)
print(zarf) #print(zarf) #

zarf = calculation_stresses(1000000)
print(zarf)

#[5.0, 6.666666666666667, 7.5, 1.2, 1.0, 5.0, 22.5, 20.0]






#------------- Function haro baham mibandim
#1- --> local variables hastan 




def newton(m,a):
    f = m * a
    return f

#newton(m=10,  a=5)
#zarf = newton(10,5,454334)
zarf = newton(10,5)

print(m) #name 'm' is not defined
print(f)  #NameError: name 'f' is not defined

'''
tasbe tarif shode , input body otuput
call-->
zarf bsaz 
newton() 
newton hast? bale --> ... not defined
newton(2 ta vorodi migire) -->ag dota vroodi ddi ejra , error
m=10 , a = 5 do ta zarf misaze
f = m*a = 50 -->sevomin zarfo

return f --> return 50 --> 

zarf = newton(10,5)
zarf = 50 

'''


def newton(m,a):
    global f #zarfe f --> nashkoneshg-> sarasari besazesh
    f = m * a
    return f



def newton(a):
    pass

def euler(a):
    pass




#-----------

def newton(m,a):
    f = m*a
    print(f)
    return f


zarf = newton(m=10,a=5) #50


zarf = newton(m=10) 
#TypeError: newton() missing 1 required positional argument: 'a'


#ag yeki a ro nadad, defautl dashte bashe
#a oo dadi ono mizrm
#ag nadadi default

#default sazi

#oon vorodi (argument) --> default
#def name(arg=default)


def newton(m,a=9.8):
    f = m*a
    print(f)
    return f


newton(10,5) #-->m=10 , a= 5 --> 50 

newton(10) #-->m=10 a=9.8 -->98


#tavabe b ch dardi mikhoran??


#--entehaye python basic , foundation


#class --->



'''
Class madar baraye function

niazi nis ma bdonim chie --->
class in python


print()
scan <<
{}


C  


C++



frghe asli --> C++ --> class ha
C ono nadare


python --> class ha support


gahi shoma nmitoni az tavabe estefade koni


class besazi



1- Psudocode
2-Function based(micro service) 
3- Object oriented programming (Class, object)




'''



class name:
    
    def __init__(self,name):
        pass
    
    def calculate(self,name):
        pass
    
    
    def stress(self):
        pass
    
    
'''
class --> variables, functions daran




a= model()

#model --> koli tabe dare



a.calculate

'''

class model:
    
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass
    
    
    
#a = model()

#a.fit()
#a.predict()
#a.felani()....



#str , list , set ,---> class

#object mnisazid
a='ali'


#class seda mzijni zarf --> object
a = str('ali')


#method hash estefade (function ha)
a.lower()
a.upper()




#===========================

'''
az y tabe too ye code dgat
biay inja estefade kon


ampro.py 

L4.py estefade konm



import ampro








'''





#ampro.py --> reperesntation -->ampro


#boro too ampro tabeye felanio bekesh biron


#ampro.esme_tabe



#zarf = newton(mass = 10 , acceleration = 5)
#kole file ro import koni
#oroye file.newton()
import ampro

zarf = ampro.newton(mass=10, acceleration=5)

print(zarf) #50



#from .py import function
#az file .pu felan function ro biar
from ampro import newton

zarf = newton(mass = 10 , acceleration = 5)

print(zarf)



#-------  3 halat


#halat 1 -->kole file ro bairi
import ampro


zarf = ampro.VED_calculator(power=10, speed=300, hatch=2, thcikness=4)
zarf = ampro.VED_calculator(10,300,2,4)

print(zarf) #0.004166666666666667



#ravehse 2
#az file ha kh bozorgan , nemisarfe kole fiel ro import
#boro done done 

from ampro import VED_calculator

zarf = VED_calculator(10,300,2,4)



#ravehse 3 szamani k esm yejorie ya gtoolanie , gharardadie developera


from ampro import VED_calculator as ved


zarf = ved(10,3000,2,4)


print(zarf) #0.0004166666666666667



'''

Ketabkhone --> Library

ketabkhone --> 100 ta 1 million file .py --> .py --> hezaran hezaran function , class


python --> 4,5 hadeghal - 100 ta dare kochiakn sade 


AM
Nano


Numpy --> mohasebat
sympy , scipy --> mohasebate differential 

Pandas --> kar ba data (excell)
matplotlib -> dasteton rams haye khafani 


Sklearn --> machine learning 


pytorch , tensorflow , keras --> deep learning





#1---> sklearn ro dashte bashi --> 


#2---> 
from sklearn.ai_models.linear_models import linear_regressor

linear_regressor(x,y) --> 



ketabkhone khode python dare math


'''



import math



math.sqrt(100) #10.0

math.sin()

math.cos()

math.tan()

math.tanh()

math.log()


math.log10()

math.erf()


math.pi

math.e


'''

Math 
random 
harf miznim



Numpy , pandas 








#-------------------------------
Jalasse ayande yehafte
code haee k neveshte shdoe ro zahmat bekshid run koni

jalaseye ghabli 8 ta Quiz (tamrin AM)



10 ta quize dg mizarma (fucntion benevisid)


18 ta rahta jolotin 




#----> Numpy bebinid chie??
khodeton mitonid numpy ro yad bgirid???
try konid

jome yad bgirid

Numpy chie ?? ayd

gpt, documentation ,....

4 ta ketabkhoone




10 Quiz-------------------


1- Yek listi az forces ha darim , yek listi az distance ha besazid 
bad yek threshold baraye maximum allowable strss .
berif iteration konidi too lista , stress ro hesab konid ag kamtar bood printeshon konid


2- hamon kare bala , ama age kamtar bood, berizideshon too ye list

3- hamon kare bala, ama age kamtar bood , fght beshmorid tedadesho



4- hala kare bala ro dar yek tabe besazid k y listi az force bgire , listi az distance va yek allowable stres
yani tabe 3 ta vorodi dashte bashe
khorojie in tabe hichie fght print kone har stressi k kamtar az allowable has


5- in tabe 3 ta vorodi dare ama dar khoroji tedade stress hae k kamtar hatsan ro bede


6- yek tabe benvisid k behsh densiity bedid va age balaye 98 bashe True pas bde, ag kamtar False pas bde


7- yek tabe besazid k temperature begire va ag balaye 2000 bashe False pass bde , ag kamtar bood True pass bde


8- yek tabe benvisidi k VED ro hesab kone yani vorodi begire va khoroji ved ro bde

9 - yek tabe benvisid yek listi az mass ha va volume haro bdid va behtoon dar khoroji
yek adad bede k tedade sample haee k density balaye 0.9 hast

10 - yek tabe benvisid yek listi az mass, yek listi az volume ha bdid va bre bgrde
va darsade takhalkhole hamaro hesab kone ( 1 - density * 100 ) badesh onae k 
kamtar az 5 % takhalkhol drn ro dar yek list brize va list ro bargardoone



'''






