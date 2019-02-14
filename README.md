# ComE_plus

Implementation of the paper Embedding Both Finite and Infinite Communities on Graphs

The implementation is base on Python 3.6.1 and Cython 0.25

The core algorithm is only written in Cython, so we provide a miniconda environment file to run our code.

In order to compile the Cython code run the following command:

```
python cython_utils.py build_ext --inplace
```