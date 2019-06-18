---
layout: article
title: A machine learner's musings of evolution of data sturctures and algorithms
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

So I am preparing myself for the standard coding interviews. This is the first time I am doing so in a formal manner since I started my journey in machine learning almost three years ago. The interview for my current job was for a researcher role and involved mostly machine learning and very little of classic coding interviews. So I didn't really study data structures and algorithms beyond an intuitive grasp of the fundamental ideas. It may be worth mentioning that I never passed fundamental computer science courses besides a course on programming since my background is in engineering (mechanical/mechatronics/biomedical). 

This time around, I am going to dig deeper due to the market being a bit different and demanding coding interviews as well. I believe that having been involved in cutting edge machine learning for the past two years, I have a different lens to view algorithm and data structures this time. I want to think about and imagine how machine learning will evolve classical ideas in algorithms and data structures. It should be fun!

# Types:

- counting bits that are one in an integer:
    + point: bit-wise operations perform the operation on all bits of the two numbers starting from their lowest bits.
        * ^ is XOR (returns 1 if only one of input is one).
        * ~ flips the bits of two's complement representation of a number and interprets the result in two's representation integer. Therefore, for an integer $$x$$, it returns $$-x-1$$ .

    + two's complement of a number, complements it to its defined 2^n precision when added up. If we flip the bits of the binary number and add one, we get its two's complement. 
        * Two's complements are used to represent positive and negative numbers in binary in a computer. Given a precision, the first half of binary numbers are used to represent positive numbers (most significant bit 0) and the second half of numbers are used to represent negative numbers (most significant bit 1). 
        * Using two's complement to represent signed numbers has the benefit that same addition operation can be used for both positive and negatives. 
    + integers in 'Py3' are unbounded and function of available memory.
    + Word size is stored in 'sys.maxsize' for integers and 'sys.float_info' for floats.

```python
def bit_counter(x):
    counter = 0
    while x:
        counter += x & 1
        x >>=1 
    return  counter
```

