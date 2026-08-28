#Halle Rose 8/2026
#Machine Learning Project 1: Theropods
#Based on data from "Dinosaur Facts and Figures the Theropods and Other Dinosauriformes"
#Written by Ruben Molina-Perez and Asier Larramendi
#Illustrated by Andrey Atuchin and Sante Mazzei
#Model: Can we forecast the body mass of a theropod based on its known hip height or length?

import pandas, seaborn, matplotlib.pyplot as plt
from sklearn.impute import KNNImputer as knn

df = pandas.read_csv("Theropod_Statistics.csv")
print(df.info())

#Plot variables
df.hist(bins=10)
plt.clf()

#Graphical correlation matrix
corr = df.corr()
print(corr)

#Plot heatmap
seaborn.heatmap(corr, annot=True)
plt.clf()

#Any missing values?
print("\nMissing values")
print(df.isnull().sum()) #4 in column 1

#Remove Hip height in m and observe changes to model
df_2 = df.drop("Hip height in m", axis=1)

corr_2 = df_2.corr()
print("\nCorrelation matrix")
print(corr_2)

seaborn.heatmap(corr_2, annot=True)
#plt.show()


###HANDLE MISSING VALUES###
#First, make a copy
df_3 = df.copy()

#Next, extract columns with at least one missing value and update dataframe
column_name_list = [col for col in df_3.loc[:, df_3.isnull().any()]]
print(column_name_list)

df_3 = df_3[column_name_list]
print(df_3)

#Then, impute missing values with KNN
imputer = knn(n_neighbors=3)

#Fit and transform the model
imputer.fit(df_3)
array_values = imputer.transform(df_3)

#Convert to dataframe with appropriate column names
df_3 = pandas.DataFrame(array_values, columns = column_name_list)
print(f"\ndf_3 after knn\n{df_3}")

#Check for missing values
print("\n")
print(df_3.isnull().sum()) # None

#Overlay the imputed column over the old column with missing values
for column_name in column_name_list:
    df[column_name] = df_3.replace(df[column_name], df[column_name_list])
print("\ndf")
print(df)

#Check for missing values
print("\n")
print(df.isnull().sum())

###DONE###

#Combine highly correlated features and drop the rest
df['Length / Hip height in m'] = df["Length in m"] / df["Hip height in m"]
df = df.drop("Length in m", axis=1)
df = df.drop("Hip height in m", axis=1)
print("\n")
print(df.info())
###DONE###

#Explore how the heatmap and correlation matrix has changed
plt.clf()
corr_3 = df.corr()
print(corr_3)
seaborn.heatmap(corr_3, annot=True)
plt.show()