
"""
In The Name of GOD


 Dec 12 2025

@author: Ali Pilehvar Meibody


L3-----


"""


'''

#------------

10 - based --> 0 1 2 3 4 5 6 7 8 9 namadi nasakhtan

donyaye dg --> 2  - base

0 1 ,..... - - 

lamps --> har adadi ro nmayesh bdim

lamps -> nanometeriu --> transistor (0,1)
Machine [computer, ....] --> transistor core (kernel)

zabane machine --> 0,1 (binary)


programming --> human command to machine



Human (EN) ---------> Machine(0,1)


Human (EN) <---|interface|----> Machine(0,1)


interface--> python .......



python --> langauge

1- Python built in functions --> narenji
print() input() len() type() open() ......
function --> kari anjam midan
print() -->namayesh mdie
input() -->vorodi migire
len() -->andaze ro mige
type() -->type ,....



2- Keywords ---> logic ro beham bzni 
Loogice python--> codo run mikon
az bala b paeen , chap b rast run mizane
man bkham in logico beham bnz
banafsh 


---unreserved --> Sefid
--> esme zarf 
3-variables (zarf , moteghayer)

esm = value(meghdar)

3.1. Numbers (int, float, complex,....) ** * / + - ....
XXXXa + b = c
c = a + b 
c = a * b 
-->dastor
a ro b alave b bokonn , beriz too c

a == b ? --> True false
!=
>
>=
<
<=

3.2. Boolean ( True , False)

3.3. string (str) --> 'ali'  '40' yek chand character az keybaorde shoma

    3.3.1. Zarf[index]  name[0] --> a
    3.3.2. zarf[start:end+1] --> li
    3.3.3. Str functions --> lower() . upper()..
    zarf.lower() --> emal nmishod , khorojhi mdiad
    
    new_zarf = zarf.lower()
    .upper() 
    .count('a') --> 4,5
    .find('a')
    .islower()
    
    
3.4. Iterables --> mikhaym dar yek zarf chanta value brizim

'''

#3.4.1. list

a=[10,20,30,40,50]

b=['ali',10 , 10.3223 , 1j , True]


#access
#zarf[index]

b[3] #1j

b[2:5] # #[10.3223, 1j, True]


#change --> acces change

b[3] = 800000
print(b)
#['ali', 10, 10.3223, 800000, True]

#list function --> zarf.function()

#function --> emal mishan , khoroji nmidan
#man zare jadidi bgirm



a= [10,20,30,40,50]

a.insert(1,200)

print(a) #[10, 200, 20, 30, 40, 50]



a.append(100000)

print(a) #[10, 200, 20, 30, 40, 50, 100000]



a.remove(10)

print(a) #[200, 20, 30, 40, 50, 100000]

a.pop(1)

print(a) #[200, 30, 40, 50, 100000]

a.sort()
print(a) #[30, 40, 50, 200, 100000]


a.reverse()

print(a) #[100000, 200, 50, 40, 30]


#delte ! = Clear

#delet --> vase hamishe hazf mishe
#cleear --> zarf hazf nmsihe, mohtaviate dakhelesh hazf msihe

a.clear()
print(a) #[]


del a
print(a) #NameError: name 'a' is not defined


'''
ITERABLES

list :  [10,20,30,] ordered(index) , changable, allow duplicated
    
tuple : (10,20,30)  ordered(index) , unchangable, allow duplicated
    
set :  {10,20,30} unordered(no index) , unchnagbale, no duplicated
    
    
dictionary:

    
index  value
...    ...
...

zarf[index]



key value
esm   ali
ghad    felan
vazn ...

zarf['esm']
zarf['ghad']

'''

a=['ali',40000 , 28 , True]


a[0] #'ali'



a= { 'name': 'ali' , 'money' : 40000 , 'sen' : 28 , 'master':True}


a['name'] #'ali'
a['money'] #40000



'''

1-Python built in fucntions
2-Keywords
3-Variables (numbers,bool , str , iterables) [ Data Types]



further information:
Data Types : 
    python built in functions
    python documentation list
    python documentation tuple
    python documentation set
    python documentation dictionary
    
    python list function ....
    
    
   


'''



#------- SHARTI --------------



print('salam')


'''

python vaghty rooye run mzini
az bala b paeen
az chap b ras
done done code ro run mizne


man gahi mikham in code ro 
yek vaghtaee run bzne
yek vaghtaee run nzne

--> shartish konm 
'''


name = input('namet chie?:')

print(name)


#nokteye dg --> input --> str pas mide hat ag adad bdi


sen = input('senet chande?:')


print(type(sen)) #<class 'str'>


#20
#'20'

#ali / 1400 ?

#nokte ; onjaee k mikhay az user adad bgiri 
#poshte input , int ya float

sen =  int( input('senet chande? :'))


print(type(sen)) #<class 'int'>

old_sen= '20'

sen = int(old_sen)


#------------------


#sen = int(input('senet chande?:'))

sen = 1400


print('salam')






'''
miad y zarf misaze bnema sen = 10 
print ('salam')

na agah man mikham
fght in tike az code
baraye sene 18 b bala run bshe

'''



'''
if --> agar

ghable oon khati k mikhaym
ejra nshe hamishe (shartish konim)


if shart:
    print('salam')


if shart:
    print('salam')
    


shart --> yechiz bashe
javabesh True , False


a == b
a > b
a < b
 

if shart:
    print('salam')
    


shart --> True --> print('salam')
shart --> False --> mide badi 

'''

sen = 20


print('salam')


'''
sen = 10    salam

sen = 20 salam
'''


sen = 20

print(sen == 20 ) #True
#aya sen barabar hast ba 20? True False
print( sen != 20) #False

print( sen > 10) #True

print( sen < 10) #False

# ==  !=  > >= < <= --> True False


#sen > 18 ---> nesbat b sen True False

#yechizi has k javabe True False

#joloye if bzaram
sen=20
#print(sen>18) #False
#print(sen>18) #True

if sen>18 :
    print('salam')


#agar javabe [[sen>18]] -->True shod --> print('salam') ag nshod hichi ejra nshe






sen=20

if sen>18:
    print('salam')

'''
sen=10 -->hichi print nashod
sen = 20 --> salam
'''


'''
Yek rahzan

joloye karavan migire

yeseriaro majbor mikone yekari konan



shart True ->joloshon mgiire , print('salam') ejra kon



'''


'''
sakhtare shart


if shart:
    dastooor1
    dastoor2
    dastoor3
    dastoor4
    .....

birone
    
    
    
**shart --> True False (== != > < >= <=)
dastoor (4 space, y tab jolo bashe) body (badane)
badaneye if e 

** --> man mitonm too badane , madami k 4 ta space jolom dastoor bnvisM?


** na tanha ye khat code, balke mitoni yek bakhshi az codet ro sharti sazi koni


chanta shart??/


'''



sen = 20

if sen>18:
    print('salam')
    
    
print('khodafez')


'''
sen = 10 ---> khodafez


sen = 20 -->salam khodafez



'''


sen = 40

if sen>18:
    print('salam')
    print('khosh oomadid')
    a=40
    b=a*10
    c=b/40
    print(c)



print('edame (biroone)')


#------------------
'''

age senesh > 18  ghadesh > 150

dota shart 


aya manzoret ine bayad har dota shart baham ejra shan ? 


ya mige hadeaghal yeki az in ha


'''


'''
1-hatman har dota shartam baham True she
2-Ya bgi yeki hadeghal az shrtam true she



1- hatman , dotash , sharte 1 VAAAA sharte 2 --> AND
2- hadegjal yekish , sharte 1 YAAA sharte2 --> OR


bish az yek shart emal koni AND , OR

'''

sen= int(input('senet chande?:'))
ghad= int(input('ghadet chande?:'))


if sen>18 and ghad>150:
    print('salam')

#ham bayad sen > 18 , hgam baayd ghad >150
#ham bayad shart1 True ham sharte 2

'''
sen 20 ghad 180 salam
sen 10 ghad 180  ...
sen 20 ghad 150  ...
sen 10 ghad 150 ...

'''

if sen>18 or ghad>150:
    print('salam')

'''
sen 20 ghad 180 salam
sen 10 ghad 180 salam
sen 20 ghad 150 salam
sen 10 ghad 150 ...

'''

'''

If --> shoma mikhay y ghesmat az codeto sharti koni



if shart:
    dastoor1
    dastoor2 
    dastoor3
    ....
    
dastoorate_biroon



Notes:
    1-shart bayad True False (== > < ..)
    2-fght dar soorati k True bashe ejra mishe
    3-dastoo fght print nis, .......
    4- ta madami k 4 spac (tab) --> too body -->shartian
    5- birone if --> jaee -->mostagghel
    6-mitoni chanta shart bzari
    
ham bayad hsarte 1 True ham sharte 2

if shart1 and shart2:
    dastooor
    dastooor
    dastoor
    
    
hadeghal yeki,  sharte1 true she YAAA sharte 2
if shart1 or shart2:
    dastooor
    dastrooor
    dastooor





#-----------
40 bar yechizi nrun mikonid
va migid agha
accuracy --> 50 %
40 % , 35

ag balaye 80 % accuracy shod zakhirash kon



if accuracy>80:
    ....save()


....save()


'''


'''

Family IF --> khanevade hastan
3 no IF


1- Just IF  ---------------

2- IF else --------------

3- IF elif ---------------


yad dehi , ag codio didid bbinid k ahan yanichi


'''




'''

1-Just if --> alan dars dadsam
Rahzan --> yeseri afrado moigir emige inkaro bokon (run)

ag True shod --> Inkaro bokon
ag false shod ? --> bikhial beman che ....




dorahi baz konid
ag True shod --> kare 1
age False shod --> bikhial (kare 2 ro bokon)



'''



sen = 10

if sen > 18:
    print('salam')




'''
sen = 20 --> sen>18 =True --> salam
sen = 10 --. sen>18 = False --> ..
ag shart True shod plan daram
onae k false shodan rad shan brn
'''

'''
ag True shod kare 1
age False shdo (nmigi bikhial) kare 2
dora


---------1-JUST IF ----------

        |
       shart
    -------------
    |True       | False
  kare1         




-------2-IF ELSE-------------

       |
       shart
    -------------
    |True       | False
  kare1         kare 2





if shart:
    dastoor1
else:
    dastooor2


'''


#----1- just if

sen = 10

if sen>18:
    print('salam')

'''
sen = 10 -->khali
'''



sen=10


if sen>18:
    print('salam')
else:
    print('mamnooo')



'''
sen>18 -->True False 


10 > 18 --> False

10 --> mamnooo

'''


'''
1--->rahzan amal koni gharbalgar
beyen kole accuracy ha yekno ro bekeshi kenar
--> Just if



2---> dorahi baz koni
ag javb  test 100 --> print moafgag
age nashdo --> print kon namoafagh





if shart:
    dastoorr1
    dastoorat1
    dastoorate1
    dastoorate1
    
else:
    dastoorate2
    dastoorate2
    dastoorate2




'''
'''
       |
       shart
    -------------
    |True       | False
  kare1         kare 2


'''

if sen>18:
    print('salam')
    print('khpsh pmadid')
    a=10
    b=a*10
    
else:
    print('bebakhshid')
    print('shoma nmitonid vared shavid')
    a=123093278
    b=a+10
    

'''

1-python built in functions

#--------------------
2-keywords-----
2.1. Shoroot
   2.1.1.Just If 
   2.1.2. IF else
   2.1.3. if elif else

2.2. Loops
    2.2.1. For
    2.2.2. While
    
2.3. Functions

#--------------------

3-variables


'''

'''

1- Just if --> rahzane
Treu --> ytek kar


        |
       shart
    -------------
    |True       | False
  kare1         



2- IF else --> dorahi
ag True shod kar1
ag false shod kar2



       |
       shart
    -------------
    |True       | False
  kare1         kare 2





3- if elif else --> dorahi haye too dar tooye


       |
       shart1
    -------------
    |True       | False
  kare1         shart2
                 |
                ------
               |       | 
             kare1     kare2





       |
       shart1
    -------------
    |True       | False
  kare1         shart2
                 |
                ----------
               |True     | False
             kare1     Sharte3 
                         |
                         





'''

#1-----Just if

sen = int(input('senet cheghadre?:'))


if sen>18:
    print('khosh oomadid')
    
#fght batraue balaye 18 ye code ro ejra mikone
#hichi b hcihi



#2---- do rahi if else

sen = int(input('senet cheghadre?:'))

if sen>18:
    print('khosh ommadid')
else:
    print('seneton ghanoni nist')
    
    
#3--- if elif else

sen = int(input('senet cheghadre?:'))

if sen>18:
    print('khoosh omadid')
elif sen>15:
    print('nazdiki b ghanon')
else:
    print('ghanoni nisti')

'''

         |sen>18
    --------------------------
    |True                     | False
     
   khosh omadid               |
                             sen>15
                ----------------------
                |True                |False
                nazdiki          ghanoni nisti

'''




'''

Txt1 -->
1- mashin hesab besazi
numb1 = input()
numb2 =
operator = 

agar operator -> jam  numb1 + numb2 print
zarb -->
taghsim




Text 2 -->
sample
density
tool=lpbf ,..

agar density balatar 0.98 
begid quality control passed
dar gheyre insoorat ebegid pass nis



3 -->
yek sample density , tool , scanning speped

density = input('density ro begid:')
density = 323

age density >0.98 pass
ag kamtar bood check konid ag scanning speed > 200 Pass , < pass nisr

'''



#==================================
sen=10
print('salam')


#shartish krdim

#logic ro beham rikhtim


#tekrar ,


print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
print('salam')
'''
salam
salam
salam
salam
salam
salam
salam
salam
salam
salam
salam
salam

'''


#yekbar minvshtm 10 bar ejra mishod
print('salam')


#manteghe python har khat ro ybar run mizne taghir bdi yejahe
#keywords

#tasavor kon keyword ---> repeat()



repeat(10):
    print('salam')

#    repeat(10):
#              ^
#SyntaxError: invalid syntax

'''
salam
salam
salam
.....
'''


'''
Loop (halghe ha)
1- For looop
2- While loop


'''



#------- FOR LOOOP-------------------
'''
shomarande --> zarf (i , j , k , esm , ali , ...)

range --> [az felan ta felan]

done done ejra kone 



'''

for i in [1,2,3,4,5]:
    print('salam')

'''
be ezaye i haee k dar in list hast , i ro gharar deh

i = 1 , i=2 , i=3 , i=4 , i=5 ....


#----
b ezaye i haee dar in list hast, i ro gharar deh va code zir ro ejra kon


i=1 print('salam')
i=2 print('salam')
i=3 print('salam')
i=4 print('salam')
i=5 print('salam')
bia biron edame bde mes ghabl





'''




for i in [1,2,3,4,5]:
    print('salam')
    
    
'''
i --> [1,2,3,4,5]
i=1 -->print('salam') --> salam
i=2 --> print('salam') --> salam
i=3 -->print('salam') --> salam
i=4 -->print('salam') --> salam
i=5 --> print('salam') -->salam



salam
salam
salam
salam
salam

'''



for i in [1,2,3,4,5,6]:
    print('salam')
    
'''
i--> [1,2,3,4,5,6]

i=1 , print('salam') --> salam
i=2
i=3
i=4
i=5
i=6

salam
salam
salam
salam
salam
salam


'''


for i in ['ali','vahid','hamid','reza']:
    print('salam')
    
'''
listi darim , be ezaye leemnt haye toye list
i ro ghrar bde , code zir(cod haee k dar body) ejra kon


i = ali --> print('salam') --> salam
i = vahid --> print('salam') --> salam
i= hamid --> print('salam') --> salam
i= reza --. Pirnt('salam') --> salam


salam
salam
salam
salam
'''




#az 1 ta 100 ejra konm chi/

#for i in [1,2,3,4,5,6,7,8,9,......,100]:
#    print('salam')
    


#tabe haye dakheli --> kolan komak mikonan bema 
#print()  len() type() open() ,....

#range()

#range(start,end+1)


#1 ta 100 --> 
range(1,101) #--> [1,2,3,4,....100]


for i in [1,2,3,4,....,100]:
    print('salam')
    


for i in range(1,101):
    print('salam')
    
    
    
'''
i --> [1,2,3,4....,100]
i=1 --> print('salam') --> salam
i=2 -->print('salam') --.> slaam

...
..
..

i=100 --> print('salam') -->salam

i = 1, ...10 --> 100 salam





'''
    

for i in range(1,101):
    print('salam')
    
    


#static repeat 
#salam 
#khodafx
#a

a=0

for i in [1,2,3,4,5]:
    b = a+1
    
    
    
'''
i=1 --> a=0 b=a+1 =0+1 = 1
i=2 --> a=0 b= a+1 = 0+1 = 1





'''
    

a=0

for i in [1,2,3,4,5]:
    a = a +1
    b = a+ 2
    
    
print(a,b) #5 7
#donbale ha , series , adad ha 
#calculation motevali


'''
i--> 1 , 2 , 3 , 4 , 5
a=0
i=1 --> a=0+1 =1 ,b=1+2=3 
i=2 -->a = 1+1 =2 , b=2+2 =4
i=3 -->a=




'''

for i in [1,2,3,4,5]:
    print(i)
    
'''

kole manteghe for loop

for --> shokmarande (i,...) tooye ye listi
dastoori ro jera kon(print() mohasebe , ....)



'''


#static


for i in [1,2,3,4,5]:
    print('salam')
    
#dynamic repeat

for i in [1,2,3,4,5]:
    print(i)
    
    
'''
i  --> 1,2,3,4,5

i=1 --> print(i) -->print(1)-->1
i=2 --> print(i) -->print(2)-->2
i=3 -_>print(i) -->print(3)-->3
i=4
i=5

1
2
3
4
5


'''

for j in [1,2,3,4]:
    print(j)
    
'''


j--> 1,2,3,4,

j=1 -->print(j)-->print(1)-->1
......

1
2
3
4

'''


for esm in ['ali','hamid','vahid']:
    print('salam')
    print(esm)


'''
esm --> ali,  hamid , vahid , code ziro hamasho baham ejra mikone


esm =alu --> print('salam') print(ali) -->< salam ali
esm = hamid --> print(salam ) print(hamid) --> salam hamid
esm - vahid --> print(salam)print(vahid)-> salam vahid

salam
ali
salam
hamid
salam
vahid


'''



for i in range(0,101):
    print(i+1)


'''
i=0 print(i+1) print(0+1) print(1)-->1
i=1 print(i+1)  print(1+1) print(2) -->2
..
.
.

i=100 print(I+1) print(101) -->101


1
2
3
4
5
6
7
.....
94
95
96
97
98
99
100
101

'''




#2 ta 2 ta beram, 3 ta beram chi??
    
    
#range(start,end+1,step) 
    
#end+1 --> 0 , end+1

#0 --> 10-1 = 9 
#
for i in range(10):
    print(i)
    

#(start , end+1)
for i in range(0,10):
    print(i)


#(start , end+1 , step)
for i in range(0,10,1):
    print(i)
    
    
#dota dota boro

for i in range(0,10,2):
    print(i)
    
    
'''
range(0,10,2)--> [0,2,4,6,8]

i=0 --> print(i)-->pirnt(0)-->0
2
4
6
8

0
2
4
6
8

'''


#--While va for hich frghi ndre


'''
While ro tadris nmikonan


mantegh
for i in range(start, end , step):
    dastoor
    ]dastoor
    dastooor
    
    
mantegh
i = start
while i<end:
    dastooor 
    dastooor
    i = i + step
    



'''



for i in range(0,100,1):
    print('salam')
    
   
'''
1 -i  2-list(start, end , step)
i --> adad ad --> ejra mikone




''' 
   


i=0
while i<100:
    print('salam')
    i=i+1

'''
i=0  , i<100 --> True --> salam , i=0+1 =1
1<100 -->salam 2 , 
2<100 -_> salam 3
3<100 


i=start mide

ta zamani k shart True hast  ejra kon code zir ro





i , ...



while SHart:
    dastoooor
    
    


'''

#yechizio bekhay bezari roo loop ta abad
#(staryt ,end)
i=0
while True:
    a=100
    b=400
    print('salam')
    i=i+1
    
    if i>10:
        break
    

#automation --> looop 
#ag felan shdo break bia biron STOP




#-------------------------

#for --> repeat , iteration ham estefade mishe


for i in ['ali','hamid','vahid']:
    print(i)


#iteration --> varresi

#ye list varresi koni done don

#yekar dg

for esm in ['hamid','ali','vahid']:
    if esm[0]=='a':
        print(esm)
#ali
'''
esm =  , hamiud ,ali, vahid --> if esm[0]=='a': print(esm)


esm = hamid --> if esm[0]  h=='a' 

esm = ali --> if esm[0]  a==a print(esm) print(ali) ->ali

esm = vahid --> if ems[0] v ==a 

'''



mylist=['vahid','hamid','ali','mohsen','amir','asal','fateme','abbas','abbas']

for esm in mylist:
    if esm[0]=='a':
        print(esm)


'''
ali
amir
asal
abbas
abbas

'''


'''

function -->kar konim

function hastan  

Numpy , Pandas, Matplotlib


Numpy --> mohasebat mdoelsazi
pandas --> dade excell ,...
matplotlib --> rasmo , rasm 


'''



'''

-----------------
Q1 -->
1- mashin hesab besazid
3 ta value begirid az karbar
numb1 = int(input('numbere 1 :'))
numb2 =
operator = 

badesh
agar operator -> jam  numb1 + numb2 print
zarb -->
taghsim



-----------------
Q2-->
sample
density
tool=lpbf ,..

agar density balatar 0.98 
begid quality control passed
dar gheyre insoorat ebegid pass nis


-----------------
Q3 -->
yek sample density , tool , scanning speped

density = input('density ro begid:')
density = 323

age density >0.98 pass
ag kamtar bood check konid ag scanning speed > 200 Pass , < pass nisr


-----------------
Q4 - yek halghe benevisid k az adade 10 ta 80 ro print kone


-----------------
Q5 -  halgheye balaro ba while benevisid


-----------------
Q6 - yek list az sample ha darim va oonro print konid

sample = [10.2 , 10.9 , 10.8 , 10.8 , 10.6 , 10.32]


------------------
Q7 - yek Machine learning ro 10 bar run zadim va bema
yek list az 10 ta accuracy dade, fght accuracy haye balaye 80% ro 
print konid

accuracy_list = [15 , 87 , 73 , 65 , 89 , 92 , 34 , 99 , 50]



------------------
Q8 - yek list az dnsity sample ha dairm , fght sample haee k 
density balaye 0.98 darand ro print konid

density_list = [0.92 , 0.9232 , 0.972332 , 0.983232 , 0.97233 , 0.983232, 0.992332 , 0.992232, 0.922323]


shoma zahmat bekeshid hal konid
tamrin nakoniu
mataleb ja nmuiofte


alipilehvar1999@gmail.com




'''
