---
layout: article
title: The amazing power of matrix factorizations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

## Algorithms design patterns
- **Recursion**: *A recursive operation is defined in terms of a variation of itself. A recursion needs to have a well-defined starting point, recursive operation, and completion conditional that doesn't depend on recursion.*

- **Divide & Conquer**: *D&C is based on recursion. A divide and conquer algorithm recursively breaks down a problem into two or more sub-problems of the same or related type, until they become simple enough to be solved directly. The solutions to the sub-problems are then combined to give a solution to the original problem.*

- **Prune & Search**: *The basic idea is to have a recursive procedure in which at each step, the input size is reduced ("pruned") by a constant factor $$0 < p < 1$$. Therefore, it is a form of decrease and conquer algorithm. If $$T(n)$$ is the computational cost of an operation with input size $$n$$, the computational cost of the operation with reduced input by factor $$p$$ is $$T(n(1-p))$$. If the cost of reduction/pruning is $$S(n)$$, then the total cost forms a recursion of the form $$T(n) = T(n(1-p)) + S(n)$$*

- **Amortization**: *Distributing the cost of an operation over many steps so as to have the cost die out and gain considerable computational efficiency. For example, repeating an operation in exponential time-intervals will amortize the cost over the total time.*

- **Dynamic Programming / Memoization**: *The task of breaking down a complex problem into a collection of subproblems, solving each subproblem just once, and storing the results. Next time the problem comes up, use the stored results instead of solving the problem again thereby trading in computational time complexity for modest space complexity. Storing solutions to problems instead of recomputing them is called memoization* 

- **Greedy Methods**: *Greedy methods follow the heuristic of making the locally optimal choice at each stage with the hope of finding a global optimum. Therefore, in essence greedy methods assume a sequence of steps involved in getting to an optimal solution.*
    + **Greedy method vs. Dynamic programming** :*A dynamic programming algorithm examines the previously solved subproblems and combines their solutions to give the best solution for the given problem while a greedy algorithm treats the solution as some sequence of steps and picks the locally optimal choice at each step.*

- **Brute Force**: *Brute force is simply solving all cases of a problem. It is the last straw, where if none of smarter/faster algorithms apply, we may use brute force to solve a problem.*

## Data Structure Patterns
- **Arrays**:
    + Stack 
    + Queue 
    + Dequeue 
- **Linked Lists**:
    + Stack
    + Queue
    + Dequeue
- **Tries**
    + tree
    + trie
    + graph
    + heap
- **Hash tables**
