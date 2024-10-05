import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

dataset=pd.read_csv("Placement_Dataset.csv")

# Basic information about the dataset
print(dataset.shape)  
print(dataset.info()) 
print(dataset.head())  

#cleaning data

#1.checking missing values
print(dataset.isnull().sum())
# as we have 67 missing values in the colum salary 
# we can replace the values or we can drop the column.
dataset['salary'].median()
#replacing median values in the missing values
dataset['salary'].fillna(dataset['salary'].median(),inplace=True)
#verifing null values
print(dataset.isnull().sum())

## missing values are removed


#2.removing duplicates values


# Remove duplicates
print(dataset.drop_duplicates(inplace=True))
# we have no duplicates values


#3. label encoding
import sklearn
from sklearn.preprocessing import LabelEncoder 
label_encode=LabelEncoder()
dataset['target']=label_encode.fit_transform(dataset['status'])
print(dataset.head())


#. here we can see the relationship betweeen the variables 
#placements count
sns.countplot(x='status', data=dataset)
plt.title('Placement Status Count')
plt.show()
#relationship betwwen salary and gender
sns.barplot(x='gender', y='salary', data=dataset)
plt.title('Salary by Gender')
plt.show()
# here we can say that males have hight  

#relationship between scc marks and salary
sns.boxplot(x='workex', y='ssc_p', data=dataset)
plt.title('SSC Percentage by Work Experience')
plt.show()
# Scatter plot to see relationship between degree percentage and salary
sns.scatterplot(x='degree_p', y='salary', data=dataset)
plt.title('Degree Percentage vs Salary')
plt.show()
# according to their percetages in degree theri salaries are placed

#corelation between the colums
correlation_matrix = dataset[['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'etest_p', 'salary']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
# from co relation matrix we can say that 
# 1.who have high employblitiy test score those have high salary.


