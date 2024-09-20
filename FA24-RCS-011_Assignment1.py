# Machine Learning First Heading
# Task 1: Lists, Dictionaries, Tuples
# **Lists**

nums = [3,5,7,8,12]

cubes = []

for x in nums:
    item = x**3
    cubes.append(item)

print(cubes)
nums.append(cubes)
print(nums)

# ### **Dictionary**

dict = {}

dict["parrot"] = 2

dict["goat"] = 4

dict["spider"] = 8

dict["crab"] = 10

print(dict)

dictItems = dict.items()

print(dictItems)

#for animal, legs in dict.items():
    #print(f"The {animal} has {legs}.")

#Sum of legs

total_legs = 0
for animal, legs in dict.items():
    print(f"The {animal} has {legs}.")
    total_legs += legs
print(f"Total Legs of all the animals: {total_legs}.")


# ### **Tuple**

A = (3,9,4,[5,6])   #tuples are immutable but we can the the array elements in because array are mutable

A[3][0] = 8

print(A)

del A

#print(A)

B = ("a", "p", "p", "l", "e")

B.count("p")

B.index("l")

# ## Task 2: Numpy

# get_ipython().system('pip install numpy')

import numpy as np

A = ([1,2,3,4],[5,6,7,8],[9,10,11,12])

A_np = np.array(A) #Numpy Array

b = A_np[:2, 1:3]

print(b)

# Empty matrix C with the same shape as A
C = np.empty(A_np.shape, dtype=int)
# C = np.zero(A_np.shape)

# Print matrix C
print(C)

z = np.array([1,0,1])

for col in range(A_np.shape[1]):
    C[:, col] = A_np[:, col] + z

print("Matrix A:")
print(A_np)

print("Matrix C (A + z added to each column):")
print(C)

#New Numpy Arrays

X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])

v = np.array([9,10])

# Add matrices X and Y
sum_matrix = X + Y
print(f"Sum of matrices X and Y: {sum_matrix}")

# Multiply matrices X and Y
product_matrix =  np.dot(X,Y)
print(product_matrix)

# Compute the element-wise square root of matrix Y
sqrt_Y = np.sqrt(Y)
# sqrt_Y = np.floor(sqrt_Y).astype(int) #Round the Decimal values and then change the datatype to integer
print(f"Element-wise square root of matrix Y: {sqrt_Y}")

# Compute and print the dot product of the matrix X and vector v
dotProduct_Xv = np.dot(X,v)
print(f"Dot product of matrix X and v: {dotProduct_Xv}")

# Compute and print the sum of each column of X
sumX = np.sum(X, axis=0)
print(sumX)

# Compute and print the sum of each Row of X
np.sum(X, axis=1)
print(sumX)


# ## Task 3: Functions and Loops

#  Create a function ‘Compute’ that takes arguments, distance and time, calculate velocity
def Compute(d,t):
    v = d/t
    return v


distance = 500
time = 8
velocity = Compute(distance, time)
print(velocity)

# Even Numbers List
even_num = []
for num in range(13):
    if num%2 == 0:
        even_num.append(num)
print(even_num)

# ## Task 4: Pandas
import pandas as pd

# Create a DataFrame
data = {
    'C1': [1,2,3,5,5],
    'C2': [6,7,5,4,8],
    'C3': [7,9,8,6,5],
    'C4': [7,5,2,8,8]
}

# print(data)
df = pd.DataFrame(data)
print(df)

# Filtered DataFrame
filtered_df = df[df['C2'] > 6]
print("Filtered DataFrame (Age > 30):")
print(filtered_df)

# Print only the first two rows
print(df.head(2))

#  Print the second column.
print(df['C2'])

#  Change the name of the third column from ‘C3’ to ‘B3’
df = df.rename(columns={'C3': 'B3'})
print(df)

# Add a new column to the dataframe and name it ‘Sum’.
# Sum the entries of each row and add the result in the column ‘Sum’.
df['sum'] = df.sum(axis=1)

print(df)

# Read CSV file
path = r'C:\Users\muham\Downloads\hello_sample.csv'
df2 = pd.read_csv(path)
print(df2)

print(df2.tail(2))

df2.info()

#Returns a tuple - first value shows the number of rows and the second value show the number of columns
print(df2.shape)  

df_Sort = df2.sort_values(by= 'Weight')

print(df_Sort)

# To check for missing values
print(df2.isnull())

# To drop rows with missing values
df_dropped_rows = df2.dropna()
print("DataFrame after dropping rows with missing values:")
print(df_dropped_rows)

# To drop columns with missing values
df_dropped_columns = df2.dropna(axis=1)
print("DataFrame after dropping columns with missing values:")
print(df_dropped_columns)





