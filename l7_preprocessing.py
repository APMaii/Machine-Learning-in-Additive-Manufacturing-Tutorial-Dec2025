#---------
'''

So we have some libraries 

time , random , ....

firstly we can go for numpy 


List --> what is list, how it can acces and change ...
Numpy --> fast  and more dimensional
numpy.ndim ,,. --> also numpy.random.randint() numpy.arrange() , numpy.zeros() ,.. numpy.reshape() 

panadas --> pandas datafame (data ) , columns , index 
pandas.read_excel() read_csv() --> data
and then we can do something like 

'''


#what can we do ?
#df.columns
#df['column]


#df.loc[]
#df.iloc[]


#
s2.abs()
s2.add()
s2.div()
s2.divide() #similar
s2.divmod() #integer
s2.multiply() #*
s2.mul()
s2.pow()

s2.pop() #remove
s2.clip()  #thresholding


s2.all()
s2.any()
s2.max()
s2.min()
s2.argmax()
s2.argmin()
s2.astype() #dtype
s2.view()
s2.copy()
s2.keys()
s2.items() #for i,j in 
s2.apply()
s2.filter()
s2.isin()
s2.isna()
s2.isnull()
s2.fillna()
s2.drop()
s2.drop_duplicates()
s2.dropna()
s2.ffill()
s2.bfill()


values = [
            [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]

df = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )



#----filtering----
df[df["birthyear"] < 1990]
df[df['age'] > 30]


#---new
#adding new column easy-----
people["age"] = 2018 - people["birthyear"]  # adds a new column "age"
people["over 30"] = people["age"] > 30      # adds another column "over 30"

df['age_plus_5'] = df['age'] + 5



#---remove
a=np.random.uniform(0,10,size=(50,3))
data=pd.DataFrame(a,columns=['Temp','Time','Modulus'])

data2=data.drop(columns='Modulus')


data.drop(columns='Modulus',inplace=True)
#inpalce=tru yani agha taghirati k goftmno roohamini k dot xadam (data ) emal kon man zarfe jhadid nmikham

zarf=data.drop(index=1) #radif ro hzzf krd
data.drop(index=1,inplace=True)


#--yadet nare
data.reset_index(drop=True,inplace=True)



#---concat
df1=pd.DataFrame([1,2])
df2=pd.DataFrame([3,4])

zarf=pd.concat([df1,df2])








people.plot(kind = "line", x = "body_mass_index", y = ["height", "weight"])
plt.show()


people.plot(kind = "scatter", x = "height", y = "weight", s=[40, 120, 200])
plt.show()





#------processing-------
df.read_excel()

df.read_csv()


print(df.info())
print(df.describe())
print(df.isnull().sum())  # check missing values
print(df.duplicated().sum())  # check for duplicates




print(df.describe())

#to include
df.describe(include='all')


df['price'].describe()





'''
Mean: df['price'].mean()
Median: df['price'].median()
Mode: df['price'].mode()
Standard Deviation: df['price'].std()
Variance: df['price'].var()
Min / Max: df['price'].min() / df['price'].max()
df['price'].skew()      # Skewness
df['price'].kurtosis()  # Kurtosis
df['price'].quantile(0.25)  # 25th percentile
df['price'].quantile([0.25, 0.5, 0.75])  # Q1, Q2 (median), Q3

Correlation Matrix: df.corr()


'''



df.corr()       # Pearson correlation between numeric columns
#Use df.corr(method='spearman') or 'kendall' for other correlation types.

df.cov()        # Covariance matrix




#----for categorial

df['category'].unique()          # List of unique values
df['category'].nunique()         # Count of unique values
df['category'].value_counts()    # Frequency of each value



import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['price'])
sns.boxplot(x='category', y='price', data=df)
sns.pairplot(df)




'''
1-empty cell   yek adad khali bashe (khataye ensani, khataye dastgah, import) NAN None
2-wrong format    #asdad bashe str has
3-wrong data  (dama ha hame balaye 0 , -10)
4-duplicated (tekrari)


dalilesh harchi mikahd bashe
ama in 4 ta mroed -->moshekel data
pas---> ag ina residgegi nashan
momekne mdoele ma k data ro migire asan run nashe, moshekl dahste bashe , accuracy paen bashe va va va....

'''







#------1-EMPTY CELL----
a=np.random.uniform(0,10,size=(50,3))
data=pd.DataFrame(a,columns=['Temp','Time','Modulus'])
data.loc[5,['Temp']]=None
data.loc[17,['Temp']]=None
data.loc[20,['Temp']]=None




data.info()

#1.1.tashkhis 
#felan ba data.info()
#empty cell haro tshkhis dadi

#sade tarin akri k mitoni koni
#bgei agha boro harjaaa harjaa empty cell has rmemoev kon oon radifo
data.dropna(inplace=True)
data.info()


#sadettarin ine ye adad khdoet bzari
data.fillna(10,inplace=True)

#pishrafte tar
#hey zarf misazam shoam inpalce=True

mymean=data['Temp'].mean()

new_data=data.fillna(mymean)

new_data=data.fillna(method='ffill') #haraj khalie gjhablairo mizare

new_data=data.fillna(method='bfill')

new_data.info()





#----------

#or fill teh one specific columns
# For numerical columns
df['age'].fillna(df['age'].mean(), inplace=True)

# For categorical columns
df['gender'].fillna(df['gender'].mode()[0], inplace=True)




#2------wrong format gahlate

#temp--.float ,int  str 


data=pd.DataFrame([['1',2,3],['2',5,6]],columns=['temp','pressure','modulus'])

data.info()



data['temp']=pd.to_numeric(data['temp'])

df['age'] = df['age'].astype(int)  # Convert to integer
df['date'] = pd.to_datetime(df['date'])  # Convert to datetime









#-----3-moshekle mantehi dre
data=pd.DataFrame([[20,2,3],[50,5,6],
                   [30,2,3],[70,5,6],
                   [20,2,3],[80,5,6],
                   [90,2,3],[100,5,6],
                   [20,2,3],[24,5,6],
                   [-10,2,3],[28,5,6],
                   [22,2,3],[20,5,6],
                   [20,2,3],[20,5,6]],columns=['temp','pressure','modulus'])

#n az format na khalie na hcihi
#khode data msohekel m,anteghi
#hala b har dalili

#aval tashkhis

#for bzni bri too tmep ha va bbini
count=0
for x in data.index:
    if data.loc[x,'temp']<0:
        count=count+1

print(count) # 12 doone damaye zxire sefrf vodjod dare

import matplotlib.pyplot as plt
y=data['temp']
plt.plot(y,'o')


#hazf
for x in data.index:
    if data.loc[x,'temp']<0:
        data.drop(x,inplace=True)

#jaygozin koni
for x in data.index:
    if data.loc[x,'temp']<0:
        data.loc[x,'temp']=20 #y amiangine ...
        

#-------
df[df['price'] > 100]

#--for multiple
df[(df['price'] > 100) & (df['stock'] > 0)]




#4------Duplicated
data.drop_duplicates(inplace=True)

#--------BAD AZ DATA CLEANING-------
#vaghty akret tamom shod
data.reset_index(drop=True,inplace=True)





#5-----encoding------
#Option A: Label Encoding
df['gender'] = df['gender'].map({'male': 0, 'female': 1})


#one hot encoding
df = pd.get_dummies(df, columns=['city'], drop_first=True)





#finally
data.reset_index(drop=True,inplace=True)


print(df.info())
print(df.head())


#-----finally save
df.to_csv('cleaned_data.csv', index=False)
df.to_excel('report.xlsx', index=False)
df.to_json('data.json', orient='records')





#-----------









