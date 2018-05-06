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
>>>import numpy as mp
>>> np.isscalar(3.1)
True
>>> np.isscalar([3.1])
False
>>> np.isscalar(False)
True
```

# Vectors

Vectors are ordered arrays of single numbers and are an example of 1st-order tensor. Vectors are fragments of objects known as vector spaces. A vector space can be considered of as the entire collection of all possible vectors of a particular length (or dimension). The three-dimensional real-valued vector space, denoted by ℝ<sup>3</sup> is often used to represent our real-world notion of three-dimensional space mathematically.

x = [x<sub>1</sub>  x<sub>2</sub>  x<sub>3</sub>  x<sub>4</sub> ... x<sub>n</sub>]

To identify the necessary component of a vector explicitly, the i<sup>th</sup> scalar element of a vector is written as x<sub>i</sub>. 



3. Matrices
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
