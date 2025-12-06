"""
Created on Sat Dec  6 16:11:23 2025

@author: Ali Pilehvar Meibody



L2 - 6 DEC 2025




Review-----------------
ADAD 10-BASED -0 1 2 3 4 5 6 7 8 9 .... --
2 based (binary , 0,1) 0 1 

Computer --> transistor (lamp) --> 0 ,1 number store, change , ....


Machine (laptob, phone ,..) --> advanced calculator


Programming --> human can command to the machine


Human (En) --------> Machine (0,1 binary)



Human (En) <---| Programming language|-----> Machine (0,1 binary)
C , C++ , Java ,.. --> 01011001 --> mokhafaf

C , C ++ , Python --> hameyze zmaine ha mhameye command
Javascript , HTML , CSS ---> web 
GO , Dart ---> Mobile 
Java , kotlin ,... --> Mobile android 


C, C++ --> Sakaht atr gahdimit ar
Python --> all things 




Computer --> 1-python 2-IDE (mohite code zani)

IDE --> spyder, vs code ,.... -->
telegram watsapp --> code mzinid
ejra -->python 
az bala b paeen
chap b rast abratoon --> compile (transalte)
rooye CPU ejra code




Python --> language --> vocab, grammar
# quot --> comment mikhone


--------Reserved words-----------
1 -Python built in fucntion ( tavabe dakheli)
narenji --> () --> yekari
print()  namayesh mide 
open()
len()
type()
id()

2-Keywords --L> banafsh , if , else, for , while
--> mantegh logice codo beham
python az bala b paeen chap b rast 
done done khataro mikhon
ye khato nakhon, y khato sharti 
khat 1000 bar bkhoon -->

--------UnReserved words-----------
3- Variables --> Sefid --> esme zarf
esme zarf (variable,moteghayer)


zarf --> esm , megdhar(value)


3.1.Numbers 
   3.1.1.INt 0,1,2,3,
   
   a=10
   b=30
   
   a=40
   
   ** * / + - 
   
   c = a + b 
   
   
   3.1.2. Float 1.32323
   
   a= 1.3232
   
   3.1.3. Complex
   
   a = 1j + 8
   
   ==
   !=
   >
   < 
   --> True False
   
   a=10
   b=30
   a>b --> True 
   
 
3.2. Boolean --> True , False
   

3.3. Str string --> reshte hastan
---> character , harf, kalame , jomle
quotation

a='salam'
name = ali --> name esme zarf , ali esme zarf

a=10
b=20
a=b

name = 'ali'

kalame -> tarkibi az keyboard haye shoma


"""

print('salam')
print     ('salam')
print'salam'
print('')

len('salam'

#tavabe( )




#------ STR---------

name = 'ali'

family = 'pilehvar'

print(family)



#---assignment-------

name = 'ali'

#tavae dakheli
print(name)

type(name) #str
len(name) #3
id(name) #6110903088

print(type(name)) #<class 'str'>
print(len(name)) #3


#------- Python built in function

print('ali')
print(40)


#print('ali') print(40)


print('ali',40) #ali 40


print('ali',40,'salam',32982379876)
#ali 40 salam 32982379876


#-->vorodi begiram az user?

ghad = input('salam gahdet chande:')

#zarfi bname ghad misaze
#jolosh input

#input --> 


#https://docs.python.org/3/library/functions.html


#abs
b=abs(-10)
print(b) #10


a = 2 **3
print(a) #8


pow(2,3) # 8


#benevsiam salam senet chande?#
#-->yechizi benevise
#man bgiram brizamesh tooye y zarf save konm



sen = input('salam senet cheghadre?')

#--------------------
#str ---->

name = 'alipilehvarmeibody'


print(name) #alipilehvarmeibody
len(name) # 18
type(name) #str
id(name) #6112676800


#slicing , indexing

#--> str --> 1 2 3 4 5 

#0 1 2 3 4 5 6 


#t e h r a n
#0 1 2 3 4 5 

#index --> neshangar
city = 'tehran'


#behehs dastresi peyda konm?
#b harfe 3 vome (indexe 2 ) dastersi
#bekeshish biron brizi too y zarfe ?

city[2]

#zarfe city , indexe 2 ro bekan biar inja

#pass dad

b = city[2]

print(b) #h


b = city[0]
print(b) #t



#slicing ---> az yeja ta yeja

name = 'alipilehvarmeibody'

name[0] #'a'



name[11] #'m'

len(name) #18 -->
#0 ...... 17

name[17] #'y'


#11 12 13 14 15 16 17 

#zarf[start:end+1]

name[11:18]

#: --> ta
# 'meibody'


#az zarfe name --> alipilehvarmeibody
#boro az index 11 ta 17 -->bekesh biron




full_name = 'alipilehvarmeibody'


last_name = full_name[11:18]


print(full_name) #alipilehvarmeibody
print(last_name) #meibody


#python 3 nokte -->


#1---> hamechi az 0 shoro mishe
#2---> range [] harchi bashe , adade akahr exclude
#3---> -1 --> not found , akhari
full_name[17] #'y'


full_name[-1] #'y'

#yeki monde b akahr?

full_name[-2] # 'd'


#az dorah mitoni b m brsi 
#yekia z chap beresim
full_name[11] # 'm'
#yeki az rast
full_name[-7] #'m'


#-----> 
full_name[-10:-7]


#----->
#step=1
#[start : end+1 : step]
last_name = full_name[11:18]
#az  az 11 ta 18 --> 1 done done biar
full_name[11:18] #'meibody'
full_name[11:18:1] # 'meibody'
full_name[11:18:2] #'mioy'
#3,4,5,.....
#[start : end+1 : step(optional)]


#-----------------------
'''
1-python built in functioon
print() input() len() type() open() ... 
https://docs.python.org/3/library/functions.html


2-Keywords --> logic
if else , elfi ,.......

3-Variables
3.1.Numbers (int,float,complex) ** * / + - == > <
3.2.Bool (True , False)
3.3.Str ( string --> 'reshte word, jomle..')
  3.3.1. Asssign name = 'ali'
  3.3.2. Change index --> zarf[index] name[4]
  3.3.3. Slicing --> zarf[start:end+1] , zarf[start : end+1 : step]
  3.3.4. STR FUNCTIONS .......
  
'''


#------------STR FUNCTION------

#fght o fght o fght baraye str ha
#omdn yeseri function sakhtan --> tabe

#lower() --> koochik mikonad oon name ro

name = 'ALI'

#---> ali

type()
len()

#lower(name)


#python built in function
#tavabe ye dakhelie kolie python

#tabe fght baraye str ha

#tabe haye narenji --> globall --> 
#str, int , float , list ,....
type(10)
type('dsdsdsds')
type(True)
type(1j)

#Lower ... alan mikham harf bnzm->str

#lower('ALI')

#esme zarf -- str mirizi
name = 'ALI'

'''
lower() upper() find() count() strip() split() 

function()



zarf.function()

'''

print(name) #ALI

#lower(name)
#str functions--->
name.lower() #'ali'

a = 10
a=10
a       =    10


name = 'tehran'

name.upper() #'TEHRAN'


name = 'in the name of god'

name.title() #'In The Name Of God'

name.capitalize() #'In the name of god'


'''
1-----> Convert mikonan
.lower() koochik mikone
.upper() bzoorg mikone
.title() title varane
.capitalize() capital mikone harfe aval
.strip() ---> space chapo rasto mizane
.replace('ghadimi','jadid')
.split(',') --> joda joda mikone


2----> adad pas mide
.count('a') --> 4
.find('a') ---> indexesho


3----> True false pas mide
.isupper() ---> True , False
.islower() --> 



'''


name = 'ali'


name.upper() #'ALI'


print(name) #ali


#tavabe --> tavabe STR -->

#1- Emali
#2- Khorojian

#str fucntion --> emali nis, khoroji deh hast

name = 'ali'
name.upper()
print(name)



name='ali'

new_name = name.upper()

print(name) #ali

print(new_name) #ALI


#---------------------------
topic = 'new methods in additive manufacturing'

corrected_topic = topic.title()


print(topic) #new methods in additive manufacturing

print(corrected_topic) #New Methods In Additive Manufacturing



#---------
#replace --->
name = 'ali'

#a --> s 

new_name = name.replace('a' , 's')

print(name) #ali
print(new_name) #sli



#-----
'''

tavabe karesho
4,5 taban


List , tuple , dictionary --> tadris

chanta adad 


if --> conditionm 
looop varesh mishim


'''



#_-------

#3-variables
#3.1.number
#3.2.boolean
#3.3.str
#esme_zarf = 'value'
#zarf[index]
#zarf[start:end+1]
#str fucntions --> tavabe hastan fght baraye str has


#list function -->
#list --> typ --> methods (fucntion) 

#ANN.tavabeye khdoesh


#type --> object
#function --> method 

#object.method()


#object --> String
#str fucntion

#object.method()


name = 'ali'

print(type(name)) #<class 'str'>

#method dkahelesh

#str methods --> str fucntion

#lower(name)
 
#name --> convert --> new_name
new_name = name.lower()
#parantez chizi nmigiran

name.upper()


name.title()

name.capitalize()



#replace
#a --> b 

name.replace('a', 'b') #'bli'



name = 'yes '
name2 = 'yes'

print(name==name2) #False

print(len(name)) #4
print(len(name2)) #3

name[3] #' '


#strip()

new_name = name.strip()


print(new_name) #yes

print(len(new_name)) #3

print(new_name == name2) #True



#--->1-taghir mdian esm --> str pass
#-->2-- adad pas mide

#count
#find
name = 'alipilehvar'

name.find('p') #3


print(name[3]) #p


name.count('a')
#2

name.count('i')
#2


#3--->True False

name.islower() #True
name.isupper() # False



'''
https://www.w3schools.com/python/python_ref_string.asp

Taklif --> berid test bznud

aya onkaro mikone ya na


STR FUNCTIONS --->
yeseri fucntion hastan k mokhtas b str hastan


zarf.function()

--> emal nmikone, khoroji mide easy
khoroji --> mirizi too zarf

in atvabe 3 no hast

1-convert
new_name = name.lower()
new_name = name.upper()
#yeseri tabe ha setting mikhan

#mese replace
name.replace()

name.replace('old' , 'new')
#beran tooye name , harja old didam --> new

new_name = name.replace('old' , 'new')


2-adad operator
name.find('a') ---> adad 
name.count('a')


3- true False 
name.islower()
name.isupper()



'''



#----> 
#3.1. Numbers (int, float, complex)
#3.2. Boolean (bool)
#3.3. Strings (str)

#zarf --> value brizm

#aya mtionm too y zarf chanta value brizm????


#---> iterables --> 

#iteration --> varresi 

#iterables --> ejazeye iteration mide bema


#iterables --> List, set , tuple , dictionary

#problem : chnata value --> tooye ye zarf

a = 'ali'



afrad = ['ali' , 'vahid' , 'hamid' , 'reza']


hamechiz = ['ali' , 1000 , 1j , 10.3232 , True]


print(afrad)


#int , float, complex, bool , str

print(type(afrad)) #<class 'list'>

afrad= list(('ali','vahid','hamid'))


a = 20
a = int(20)

b=10.3232
b = float(10.43343)

#--->dakheli
int()
float()
complex()
str()
list()
set()
tuple()
#--------

a = 8.9327327

print(a)

#casting
b = int(a)

print(b) #8

print(type(b))

c = str(a)

print(c) #--> '8.9327327'


#---------
#-----4.1.LIST------


#---4.1.1.Assign
afrad = ['ali','vahid', 'hamid' ,'reza']

afrad = list(('ali','vahid','hamid','reza'))



#----4.1.2. Access

name = 'tehran'
#t e h 
#0 1 2

#indexe 2 a name
#name[2] --> h 


#afrad = ['ali','vahid', 'hamid' , 'reza']
# -------> 0      1        2     , 3

afrad = ['ali' , 'vahid' ,' hamid' ,'reza']

afrad[1] #'vahid'


farde_morede_nazar = afrad[1]

print(farde_morede_nazar) #vahid

print(type(farde_morede_nazar)) #<class 'str'>
farde_morede_nazar[2] #'h'


#accesss -->

afrad = ['ali' , 'vahid' ,' hamid' ,'reza']



zakhire= afrad[1]


#Change 

afrad[1] = 'mohsen'

print(afrad)
#['ali', 'mohsen', ' hamid', 'reza']


#slicing access

afrad = ['ali' , 'vahid' ,' hamid' ,'reza']
# index   0        1        2        3

#esme_zarf[start:end+1]

afrad[1:4] #1 2 3 #['mohsen', ' hamid', 'reza']


#4.1.4. LIST FUNCTIONS

#functions ---> fgth baraye list has hastan

#List --> class (object)
#-->meythods

#list.methods()

#

'''

https://www.w3schools.com/python/python_ref_list.asp

index()	Returns the index of the first element with the specified value


sort()	Sorts the list
reverse()	Reverses the order of the list
insert()	Adds an element at the specified position
remove()	Removes the first item with the specified value
pop()	Removes the element at the specified position
count()	Returns the number of elements with the specified value
copy()	Returns a copy of the list
clear()	Removes all the elements from the list
append()	Adds an element at the end of the list
extend()	Add the elements of a list (or any iterable), to the end of the current list







'''

#1 --> str functions --> emali nabod , khoroji

#2---> list fucntions --> emalie , niazi b zqrf


a= [10,20,3 , 2 , 1 , 100]

a.sort()

print(a) #[1, 2, 3, 10, 20, 100]


a.reverse()

print(a) #[100, 20, 10, 3, 2, 1]




afrad = ['ali', 'vahid', 'hamid' ,'reza']


#acces --> zaerf[index]

#zarf[index] = new_value

afrad[1] = ' mohsen'


print(afrad)
#['ali', ' mohsen', 'hamid', 'reza']

#----
afrad = ['ali', 'vahid', 'hamid' ,'reza']

#mohseno inser konam 

#tabe vojod insert

#replace yadetone?

#zarf.replace(old , new)

#esme_zarf.insert()
#index , object 

#too kodom index, chi ezafe koni?

afrad.insert(2 ,'mohsen' )

print(afrad)

#['ali', 'vahid', 'mohsen', 'hamid', 'reza']


print(afrad[2]) #mohsen



afrad.remove('mohsen')

print(afrad) #['ali', 'vahid', 'hamid', 'reza']

#--> 2 rah
#1--> .remove(element)
#2-->   .pop(index)

#indexe 0 
#afrad.remove(0)

afrad.pop(0)

print(afrad) #['vahid', 'hamid', 'reza']



afrad = ['ali' ,'vahid' ,'mohsen']

hazfshode = afrad.pop(0)

print(afrad) #['vahid', 'mohsen']
 
print(hazfshode) #ali


#aval -_> afrad misze

#afrad.pop(0) --> afrad alio hazf mikone
#hazfshode




zarf = afrad.insert(2 ,'mohsen' )


print(afrad)
#['vahid', 'mohsen', 'mohsen']

print(zarf) #None


#--_> str function --> khoroji dare

#new_name = name.replace()



#->list fucntin --> khroji ndre emal

#afrad.insert()


#ag yedarsad

#zarf = afrad.insert()
#inser najam msihe , zarf --> None

#yeseri tavbe list functions -->
#ham emal mikone ham khoroji 



afrad = ['ali' ,'vahid' ,'mohsen']

afrad.pop(0)




#afrad.sort()
#afrad.reversed()
#afrad.insert(index,element)
#afrad.remove(element)
#zarf = afrad.pop(index)


#append --> 
#insert --_. tahe list ezafe mikone

afrad = ['ali' ,'vahid' ,'mohsen']


afrad.append('amir')

print(afrad) #['ali', 'vahid', 'mohsen', 'amir']



afrad = ['ali' , 'ali']

'''


list , set  , tuple , dictionary

karaye mokhtalef esteafe mishe


mese haman 


'''


a = [10,20,30,40]

b = (10,20,30,40)

c = {10,20,30,40}



#hamashon --> iterable -->chanta value mriizan

print(type(a)) #<class 'list'>
print(type(b)) #<class 'tuple'>
print(type(c)) #<class 'set'>


'''

list --> ordered (index) , changable , allow duplicated
tuple --> ordered(index) , unchangable , allow duplicated
set --> unordered (non index) , non changable , No duplciated

[]
()
{}

zarf[index]
zarf[index:]


list functions
list.function()


tuple.function()
set.function()


https://www.w3schools.com/python/python_tuples.asp

https://www.w3schools.com/python/python_sets.asp



'''

'''


3. Variables

3.1. Numbers (int,float, compelx)
3.2. Bool ( True , False)
3.3. Str (assign '' , index access , str.function())
3.4.Iterables (chantra value)
   3.4.1. List --> ordered(index) , changabel, allow duplicated
       3.4.1.1.asssign [element,element]
       3.4.1.2.Accesss name[index]  name[start:end]
       3.4.1.3.change name[index] = new_value
       3.4.1.4.list functions --> list.function()
   3.4.2. Tuple --> ordered , unchangable , allow duplicated
          (element,element,element) tuple.function()
           tooye data base --> change

   3.4.3. Set --> unordered, unchangable, No duplicated
          {element, element}
          set.functions()
          
   3.4.4. Dictionary --> List (ye tafavot)
   
   
*** str functions --> emali nabod , khoroji midad (zarf)
*** list functions --> emal mikrd, khoroji nmidad

'''




customer = ['ali', 30 , 4000 , 'turin']






case_1 = ['titanium', '10 dec' ,20 , 400 , 320 , 'lpbf']





print('damayte kare ma bood : ',case_1[])

#bekaremon nemaid karo sakht mikon

#bahse etelaat information hast

#index --> key

case_1[18]
case_1[4]


case_1['temp']

case_1['speed']


#list --> Dictionary

#key value


case_1 = ['titanium', '10 dec' ,20 , 400 , 320 , 'lpbf']

'''
List


index        value
0         ti
1         10 dec
2         20
3       400
4        320
5       lpbf


name[index]
case_1[4] ---> 320



Dictionary

key        value
material      ti
date       10 dec
mass      20
speed     400

name[key]
case_1['mass'] ---> 20

'''


case_1 = ['titanium', '10 dec' ,20 , 400 , 320 , 'lpbf']


case_1_dict = { 'material' : 'titanium'  , 'date' : '10 dec'  ,  'mass' : 20   , 'speed' :200  , 'temp': 480  ,'tool' :'lpbf'}




case_1[0] # 'titanium'

case_1_dict['material'] #'titanium'

case_1_dict['speed'] #200


#dict functions

case_1_dict.keys()
#dict_keys(['material', 'date', 'mass', 'speed', 'temp', 'tool'])


case_1_dict.values()
#dict_values(['titanium', '10 dec', 20, 200, 480, 'lpbf'])



#--------
case_1 = ['titanium', '10 dec' ,20 , 400 , 320 , 'lpbf']

case_1[0] = 'Cu'

print(case_1) #['Cu', '10 dec', 20, 400, 320, 'lpbf']



case_1_dict = { 'material' : 'titanium'  , 'date' : '10 dec'  ,  'mass' : 20   , 'speed' :200  , 'temp': 480  ,'tool' :'lpbf'}


case_1_dict['material'] = 'cu'

print(case_1_dict)
#{'material': 'cu', 'date': '10 dec', 'mass': 20, 'speed': 200, 'temp': 480, 'tool': 'lpbf'}


#chizi jadid 


case_1_dict['ref'] = 'ACS Nano 2026 wANG. chen et all'

print(case_1_dict)



'''

Data Types haro tonestim yad begirim










'''





