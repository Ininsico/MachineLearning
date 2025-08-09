
#A one dimensional Array capable of holding any data type
# s = pd.Series([1,2,3,4,5]);
# # print(s)

# A two-dimensional labeled data structure with columns of potentially different types.
# data = {
#     'Name' :['Arslan','Bilal','Jalal'],
#     'Age' : [20,18,17],
#     'City' : ['Abbbottabad','Islamabad','Jhelum']
# }
# df = pd.DataFrame(data)
# print(data)

# #reading data
# df = pd.read_csv(data.csv)
# df = pd.read_excel(data.excel)
# df = pd.read_sql('SELECT * FROM tabls_name',connection)

#exploaring data
# print(df.head())
# print(df.tail(3))
# print(df.info())
# print(df.describe())
# print(df.columns())
# print(df.shape()) get shape columns and rows 

#selecting data
# print(df['Name'])
# print(df[['Name','Age']])
# print(df.iloc[0])
# print(df.iloc[1:3])
# print(df[df['Age']>30])

# Modfiying data
# add a new column
# df['Salary'] = [70000,80000,90000]
# edit column
# df['Age']  = df['Age'] +1
# delete comlum 
# df = df.drop('City',axis =1)

#handling missing data
# print(df.isnull())
# df = df.dropna()
# df  = df.fillna()


