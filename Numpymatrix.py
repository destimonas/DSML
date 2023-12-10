import numpy as np
matrix1=np.array([[1,2],[3,4]])
matrix2=np.array([[5,6],[7,8]])
print("Matrix1:",matrix1)
print("Matrix2:",matrix2)
sum=np.add(matrix1,matrix2)
print("Addition:",sum)

sub=np.subtract(matrix1,matrix2)
print("Substraction:",sub)

mul=np.multiply(matrix1,matrix2)
print("Multiplication:",mul)

div=np.divide(matrix1,matrix2)
print("Division:",div)

do=np.dot(matrix1,matrix2)
print("Dot:",do)

squ=np.square(matrix1,matrix2)
print("square:",squ)
sq=np.sqrt(matrix1)
print("square root:",sq)

tran=np.transpose(matrix1)
print("Transpose:",tran)