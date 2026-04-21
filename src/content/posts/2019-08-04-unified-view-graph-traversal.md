---
title: "A unified view of graph traversal: BFS, Dijkstra, A\\* are the same algorithm"
description: "BFS, Dijkstra, and A* differ by one line: the data structure you pop the next node from. A worked maze example that converts each into the next."
date: 2019-08-04
tags: [algorithms, intuitions]
category: data_science
---

Breadth-first search, Dijkstra, and A\* are usually introduced as three
different algorithms on three different slides. They are not. They are the
same algorithm with three different priority functions, and you can convert
one into the next by editing a single line.

The clearest way I know to see this is to solve a maze.

## The setup

We have a grid where `0` is a wall, `1` is a corridor, `2` is the start and
`3` is the goal:

```
0 0 0 0 3 0 0 0 0
0 1 1 0 1 0 1 1 0
0 1 1 0 1 1 1 1 0
0 0 1 0 0 1 0 0 0
0 1 2 1 1 1 1 1 0
0 0 0 0 0 0 0 0 0
```

We want to know if there is a path from `2` to `3`. The structure of every
solution will be the same:

1. Keep a container of "places I still need to visit".
2. Pop one out.
3. If it is the goal, win.
4. Otherwise, push its unvisited neighbours back in.

The only thing that changes between the algorithms is *which element we pop*.

## BFS — pop the oldest

A FIFO queue gives us BFS. We visit nodes in the order they were
discovered, so the first time we reach the goal we have reached it in the
fewest hops.

```python
def has_path(grid, start):
    rows, cols = len(grid), len(grid[0])
    queue = [start]
    visited = set()
    while queue:
        r, c = queue.pop(0)          # FIFO — oldest first
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if grid[r][c] == 3:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                queue.append((nr, nc))
    return False
```

For an unweighted grid this also happens to return the shortest path
(measured in number of steps).

## Dijkstra — pop the cheapest

Now suppose each cell has a cost — walking through sand costs more than
walking through grass. BFS no longer works: the first time we reach a node
may not be the cheapest way to reach it.

The fix is one line. Replace the FIFO queue with a **min-heap keyed by the
total cost to reach each node**. Everything else stays the same.

```python
import heapq

def shortest_cost(grid, start):
    rows, cols = len(grid), len(grid[0])
    heap = [(0, start)]              # (cost_so_far, node)
    best = {start: 0}
    while heap:
        cost, (r, c) = heapq.heappop(heap)   # cheapest first
        if grid[r][c] == 3:
            return cost
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                new_cost = cost + grid[nr][nc]
                if new_cost < best.get((nr, nc), float('inf')):
                    best[(nr, nc)] = new_cost
                    heapq.heappush(heap, (new_cost, (nr, nc)))
    return None
```

That is Dijkstra. A FIFO queue is a min-heap keyed by insertion order;
swapping it for a heap keyed by cost gives you shortest-path for weighted
graphs, for free.

## A\* — pop the cheapest *plus* a guess

Dijkstra explores uniformly outwards from the start. If we know where the
goal is, we can do better by preferring nodes that *look closer* to the
goal. That is A\*.

One line again. Change the heap key from `cost_so_far` to
`cost_so_far + heuristic(node, goal)`:

```python
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    heap = [(manhattan(start, goal), 0, start)]   # (f, g, node)
    best = {start: 0}
    while heap:
        _, g, (r, c) = heapq.heappop(heap)
        if (r, c) == goal:
            return g
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                new_g = g + grid[nr][nc]
                if new_g < best.get((nr, nc), float('inf')):
                    best[(nr, nc)] = new_g
                    f = new_g + manhattan((nr, nc), goal)
                    heapq.heappush(heap, (f, new_g, (nr, nc)))
    return None
```

If the heuristic is `0` everywhere, A\* collapses back to Dijkstra. If every
edge weight is `1` *and* the heuristic is `0`, Dijkstra collapses back to
BFS. They really are one algorithm:

| Algorithm | Priority on pop                 |
| --------- | ------------------------------- |
| BFS       | insertion order                 |
| Dijkstra  | `cost_so_far`                   |
| A\*       | `cost_so_far + heuristic`       |

## Why this is worth internalising

The list of algorithms you are supposed to know grows every year. The list
of *ideas* is much shorter. Graph traversal is one idea with a parameter
(what goes in the priority) and three famous settings of that parameter.
Viterbi on a trellis is another setting. Best-first beam search in an NLP
decoder is another. Dynamic programming on a DAG is another.

Once you see the skeleton, each new algorithm is a ten-minute read instead
of a lecture.

---

*This is an old note I wrote in 2019 and never published. Reposting it here
because it is the kind of thing I want this blog to be: small ideas that
collapse a pile of named things into one.*
