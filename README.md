# linear algebra for deeplearning in python

### Deep Learning: 
Deep Learning is a subdomain of machine learning, concerned with the algorithm which imitates the function and structure of the brain called the artificial neural network.

### Why Math?    

Linear algebra, probability and calculus are the 'languages' in which machine learning is formulated. Learning these topics will contribute a deeper understanding of the underlying algorithmic mechanics and allow development of new algorithms.

When confined to smaller levels, everything is math behind deep learning. So it is important to understand basic linear algebra before getting started with deep learning and programming it.

# Scalars

**Scalars are single numbers and are an example of a 0th-order tensor**. The notation x ∈ ℝ states that the scalar values in an element in a set of real-valued numbers, ℝ. There are different sets of numbers of interest in deep learning. ℕ represents the set of positive integers (1,2,3,…). ℤ designates the integers, which combine positive, negative and zero values. ℚ represents the set of rational numbers that may be expressed as a fraction of two integers.

Few built-in scalar types are int, float, complex, bytes, Unicode. In NumPy, there are 24 new fundamental Python types to describe different types of scalars. When it comes to a graphical representation, it will be a simple point on the graph [More Info](https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.scalars.html).


*Defining Scalars and Few Operations in Python*

```python3
# In-Built Scalars
a = 5
b = 7.5
print(type(a))
print(type(b))
print(a + b)
print(a - b)
print(a * b)
print(a / b)
```


```python3
# Is Scalar Function
def isscalar(num):
    if isinstance(num, generic):
        return True
    else:
        return type(num) in ScalarType
```

```python3
>>> import numpy as mp
>>> np.isscalar(3.1)
True
>>> np.isscalar([3.1])
False
>>> np.isscalar(False)
True
```

# Vectors

Vectors are ordered arrays of single numbers and are an example of 1st-order tensor. Vectors are fragments of objects known as vector spaces. A vector space can be considered of as the entire collection of all possible vectors of a particular length (or dimension). The three-dimensional real-valued vector space, denoted by ℝ<sup>3</sup> is often used to represent our real-world notion of three-dimensional space mathematically.

<p align="center" style="font-size: 22px;">x = [x<sub>1</sub>  x<sub>2</sub>  x<sub>3</sub>  x<sub>4</sub> ... x<sub>n</sub>]</p>

To identify the necessary component of a vector explicitly, the i<sup>th</sup> scalar element of a vector is written as x<sub>i</sub>. 

In deep learning vectors usually represent feature vectors, with their original components defining how relevant a particular feature is. Such elements could include the related importance of the intensity of a set of pixels in a two-dimensional image or historical price values for a cross-section of financial instruments.

*Defining Vectors and Few Operations in Python*

```python
import numpy as np

# Declaring Vectors

x = [1, 2, 3]
y = [4, 5, 6]

print(type(x))

# This does'nt give the vector addition.
print(x + y)

# Vector addition using Numpy

z = np.add(x, y)
print(z)
print(type(z))

# Vector Cross Product
mul = np.cross(x, y)
print(mul)
```

```
<class 'list'>
[1, 2, 3, 4, 5, 6]
[5 7 9]
<class 'numpy.ndarray'>
[-3  6 -3]
```

# Matrices

Matrices are rectangular arrays consisting of numbers and are an example of 2nd-order tensors. If m and n are positive integers, that is m,n ∈ ℕ then the m×n matrix contains mn numbers, with m rows and n columns.

Matrix can be written as:
<p align="center">
![Matrix Image](https://cdn.discordapp.com/attachments/391971809563508738/442640005182128129/Screen_Shot_2018-05-06_at_4.21.08_PM.png "Matrix Image")
</p>


*Defining Matrix and Few Operations in Python*

```python3
>>> x = np.matrix([[1,2],[2,3]])
>>> x
matrix([[1, 2],
        [2, 3]])
>>> z = x.mean(1)
>>> z
matrix([[ 1.5],
        [ 2.5]])
>>> z.shape
(2, 1)
>>> y = x - z
matrix([[-0.5,  0.5],
        [-0.5,  0.5]])
>>> print(type(z))
<class 'numpy.matrixlib.defmatrix.matrix'>
```




4. Tensors
5. Matrix Addition
6. Matrix-Matrix Addition
7. Matrix-Scalar Addition
8. Broadcasting
9. Matrix Multiplication
10. Matrix Transpose
11. Matrix-Matrix Multiplication
12. Scalar-Matrix Multiplication
13. Scalar-Matrix Multiplication
14. Dot Product
