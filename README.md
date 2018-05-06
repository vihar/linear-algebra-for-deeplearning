# linear algebra for deeplearning in python

## Deep Learning: 
Deep Learning is a subdomain of machine learning, concerned with the algorithm which imitates the function and structure of the brain called the artificial neural network.

## Why Math?    

Linear algebra, probability and calculus are the 'languages' in which machine learning is formulated. Learning these topics will contribute a deeper understanding of the underlying algorithmic mechanics and allow development of new algorithms.

When confined to smaller levels, everything is math behind deep learning. So it is important to understand basic linear algebra before getting started with deep learning and programming it.

<p align="center">
<img src="https://cdn.discordapp.com/attachments/391971809563508738/442659363543318528/scalar-vector-matrix-tensor.png" width="450">
</p>

The core data structures behind Deep-Learning are Scalars, Vectors, Matrices and Tensors. Programmatically, let's solve all the basic linear algebra problems using these.


# Scalars

**Scalars are single numbers and are an example of a 0th-order tensor**. The notation x ∈ ℝ states that x is a scalar belonging to a set of Real-values numbers, ℝ. There are different sets of numbers of interest in deep learning. ℕ represents the set of positive integers (1,2,3,…). ℤ designates the integers, which combine positive, negative and zero values. ℚ represents the set of rational numbers that may be expressed as a fraction of two integers.

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
        return False
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

<p align="center" style="font-size: 22px;">x = [x<sub>1</sub>  x<sub>2</sub>  x<sub>3</sub> x<sub>4</sub> ... x<sub>n</sub>]</p>

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

**The full m×n matrix can be written as:**

<p align="center">
<img src="https://cdn.discordapp.com/attachments/391971809563508738/442640005182128129/Screen_Shot_2018-05-06_at_4.21.08_PM.png" width="250">
</p>

It is often useful to abbreviate the full matrix component display into the following expression:

<p align="center">
A=[a<sub>ij</sub>]<sub>m×n</sub>
</p>

*Defining Matrix and Few Operations in Python*

```python3
>>> import numpy as np
>>> x = np.matrix([[1,2],[2,3]])
>>> x
matrix([[1, 2],
        [2, 3]])

>>> a = x.mean(0)
>>> a
matrix([[1.5, 2.5]])
>>> # Finding the mean with 1 with the matrix x.
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

#### Matrix Addition

Matrices can be added to scalars, vectors and other matrices. Each of these operations has a precise definition. These techniques are used frequently in machine learning and deep learning so it is worth familiarising yourself with them.

```python3
# Matrix Addition

import numpy as np

x = np.matrix([[1, 2], [4, 3]])

sum = x.sum()
print(sum)
# Output: 10
```

#### Matrix-Matrix Addition

C = A + B (shape of A and B should be equal)

```python3
# Matrix-Matrix Addition

import numpy as np

x = np.matrix([[1, 2], [4, 3]])
y = np.matrix([[3, 4], [3, 10]])

print(x.shape)
# (2, 2)
print(y.shape)
# (2, 2)

m_sum = np.add(x, y)
print(m_sum)
print(m_sum.shape)
"""
Output : 
[[ 4  6]
 [ 7 13]]

(2, 2)
"""
```

#### Matrix-Scalar Addition

Adds the given scalar to all the elements in the given matrix.

```python3
# Matrix-Scalar Addition

import numpy as np

x = np.matrix([[1, 2], [4, 3]])
s_sum = x + 1
print(s_sum)
"""
Output:
[[2 3]
 [5 4]]
"""
```

#### Matrix Multiplication

A of shape (m x n) and B of shape (n x p) multiplied gives C of shape (m x p)

```python3
# Matrix Multiplication

import numpy as np

a = [[1, 0], [0, 1]]
b = [1, 2]
np.matmul(a, b)
# Output: array([1, 2])

complex_mul = np.matmul([2j, 3j], [2j, 3j])
print(complex_mul)
# Output: (-13+0j)
```

#### Matrix Scalar Multiplication

Multiplies the given scalar to all the elements in the given matrix.

```python3
# Matrix Scalar Multiplication

import numpy as np

x = np.matrix([[1, 2], [4, 3]])
s_mul = x * 3
print(s_mul)
"""
[[ 3  6]
 [12  9]]
"""
```

#### Matrix Transpose

A=[a<sub>ij</sub>]<sub>mxn</sub>

A<sup>T</sup>=[a<sub>ji</sub>]<sub>n×m</sub>

```python3
# Matrix Transpose

import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a)
"""
[[1 2]
 [3 4]]
"""
a.transpose()
print(a)
"""
array([[1, 3],
       [2, 4]])
"""
```

# Tensors

The more general entity of a tensor encapsulates the scalar, vector and the matrix. It is sometimes necessary—both in the physical sciences and machine learning—to make use of tensors with order that exceeds two.

We use Python libraries like tensorflow or PyTorch inoreder to declare tensors, rather than nesting matrices.

*To define a simple tensor in PyTorch*

```python3
import torch

a = torch.Tensor([26])

print(type(a))
# <class 'torch.FloatTensor'>

print(a.shape)
# torch.Size([1])

# Creates a Random Torch Variable of size 5x3.
t = torch.Tensor(5, 3)
print(t)
"""
 0.0000e+00  0.0000e+00  0.0000e+00
 0.0000e+00  7.0065e-45  1.1614e-41
 0.0000e+00  2.2369e+08  0.0000e+00
 0.0000e+00  0.0000e+00  0.0000e+00
        nan         nan -1.4469e+35

[torch.FloatTensor of size 5x3]
"""
print(t.shape)
# torch.Size([5, 3])
```

*Few Operations on Tensors in Python*

```python3
import torch

# Creating Tensors

p = torch.Tensor(4,4)
q = torch.Tensor(4,4)
ones = torch.ones(4,4)

print(p, q, ones)
"""
Output:

1.00000e-45 *
  0.0000  0.0000  0.0000  0.0000
  9.8091  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
[torch.FloatTensor of size 4x4]

 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00
 7.0065e-45  1.1614e-41  0.0000e+00  2.2369e+08
 0.0000e+00  0.0000e+00  1.4349e-42  1.4349e-42
        nan         nan -6.2183e+37         nan
[torch.FloatTensor of size 4x4]
 
 1  1  1  1
 1  1  1  1
 1  1  1  1
 1  1  1  1
[torch.FloatTensor of size 4x4]
"""

print("Addition:{}".format(p + q))
print("Subtraction:{}".format(p - ones))
print("Multiplication:{}".format(p * ones))
print("Division:{}".format(q / ones))

"""
Addition:
1.00000e-44 *
  0.0000  0.0000  0.0000  0.0000
  2.1019  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
[torch.FloatTensor of size 4x4]

Subtraction:
-1 -1 -1 -1
-1 -1 -1 -1
-1 -1 -1 -1
-1 -1 -1 -1
[torch.FloatTensor of size 4x4]

Multiplication:
1.00000e-45 *
  0.0000  0.0000  0.0000  0.0000
  9.8091  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
[torch.FloatTensor of size 4x4]

Division:
1.00000e-44 *
  0.0000  0.0000  0.0000  0.0000
  1.1210  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000
[torch.FloatTensor of size 4x4]
"""
```

