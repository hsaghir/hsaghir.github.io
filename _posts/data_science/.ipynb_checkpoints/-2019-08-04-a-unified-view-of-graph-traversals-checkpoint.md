---
layout: article
title: A machine learner's musings of evolution of data sturctures and algorithms
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

So BFS and DFS are simplest ways of traversing a graph. Dijkstra, and A* are more sophisticated versions designed for weighted graphs where edges have costs associated with them. But in essence they are all the same and I want to show this by writing code to find a path through a maze using a simple BFS, and then do minor changes on the code to convert the algorithm to Dijkstra/A* to show how close they conceptually are:


```python
# a class for implemneting a BFS to find a path through a maze

 # You are asked to desing a function that determines  
 # whether a given maze is "solvable" or not. For example, 
 # consider the following maze: 
  
 #^I0 0 0 0 3 0 0 0 0  
 #^I0 1 1 0 1 0 1 1 0  
 #^I0 1 1 0 1 1 1 1 0  
 #^I0 0 1 0 0 1 0 0 0  
 #^I0 1 2 1 1 1 1 1 0  
 #^I0 0 0 0 0 0 0 0 0  
  
 # here values equal to 0 represent walls, values greater 
 # or equal to 1 represent corridors. Your function should  
 # return True if there is a path between the position with  
 # value 2 and the position with value 3. For the maze above, 
 # one possible solution is: 
  
 #^I0 0 0 0 3 0 0 0 0  
 #^I0 1 1 0 ↑ 0 1 1 0  
 #^I0 1 1 0 ↑ ← 1 1 0  
 #^I0 0 1 0 0 ↑ 0 0 0  
 #^I0 1 2 → → ↑ 1 1 0  
 #^I0 0 0 0 0 0 0 0 0  
  
 # For example, given  
  
 maze = [ 
     [0, 0, 0, 0, 3, 0, 0, 0, 0], 
     [0, 1, 1, 0, 1, 0, 1, 1, 0], 
     [0, 1, 1, 0, 1, 1, 1, 1, 0], 
     [0, 0, 1, 0, 0, 1, 0, 0, 0], 
     [0, 1, 2, 1, 1, 1, 1, 1, 0], 
     [0, 0, 0, 0, 0, 0, 0, 0, 0] 
 ] 
  
  
  
 # your function should return True: 
 def hasPath(graph, m, n): 
     maze_m , maze_n = len(graph), len(graph[0]) 
     queue = [(m,n)]; visited = {} 
     while queue: 
         node = queue.pop(0) 
         m, n = node 
         #import ipdb; ipdb.set_trace() 
         if visited.get(node) is None: 
             visited[node] = True 
             if graph[m][n] == 3: 
                 return True 
             for i in [-1, 1]: 
                 if m+i>=0 and m+i< maze_m and graph[m+i][n]> 0: 
                     queue += [(m+i, n)] 
                 if n+i>=0 and n+i< maze_n and graph[m][n+i]> 0: 
                     queue += [(m, n+i)] 
     return False 
```
 
 To convert this to Dijkstra, we just replace the regular queue with a priority queue i.e. a heap data structure. 


```python
# a class for implementing a Heap data structure
 class Heap(object): 
     def __init__(self): 
         self.keys = [0] 
         self.values = [0] 
     def size(self): 
         return len(self.keys) - 1 
          
     def heapify(self, keys, values): 
         self.keys += keys
         self.values += values
         for i in reversed(range(self.size()//2)): 
             self.percDown(i) 
              
     def percDown(self, i): 
         while i<self.size(): 
             minChild = self.minChild(i) 
             if minChild != None and self.values[minChild] < self.values[i]: 
                 self.keys[minChild], self.keys[i] = self.keys[i], self.keys[minChild] 
                 self.values[minChild], self.values[i] = self.values[i], self.values[minChild] 
                 i = minChild 
             else: 
                 break 
              
     def minChild(self, i): 
         if i*2 > self.size(): 
             return None 
         elif i*2 == self.size(): 
             return 2*i 
         else: 
             return 2*i+1 if self.values[2*i+1] < self.values[2*i] else 2*i 
              
     def addNode(self, key, value): 
         self.keys += list(key) 
         self.values += list(value) 
         self.percUp(self.size()) 
          
     def percUp(self, i): 
         while i > 0: 
             if self.values[i//2] > self.values[i]: 
                 self.keys[i//2], self.keys[i] = self.keys[i], self.keys[i//2] 
                 self.values[i//2], self.values[i] = self.values[i], self.values[i//2] 
                 i = i//2 
             else: 
                 break 
                  
     def popTop(self): 
         self.keys[1], self.keys[-1] = self.keys[-1], self.keys[1] 
         self.values[1], self.values[-1] = self.values[-1], self.values[1]  
         topkey = self.keys.pop() 
         topvalue = self.values.pop() 
         self.percDown(1) 
         return topkey, topvalue 
```

```python
# test the heap
h = Heap()
data ={(0,):43, (1,):12, (2,):45, (3, ):123, (4,):5123, (5, ):54, (6,):1, (7,):67, (8,):13:, (9,):98, (10,):3, (11,):2, (12,):4:, (13,):6}
h.heapify(data.keys(), data.values())
```

