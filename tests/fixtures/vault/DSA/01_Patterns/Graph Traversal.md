---
type: pattern
status: stable
tags: [dsa/pattern, dsa/graph-traversal]
canonical: true
related: [[[DFS]], [[BFS]], [[Union Find]]]
---
# Graph Traversal

Visit every reachable vertex exactly once.

## Mental Model

A traversal is a frontier plus a visited set. [[DFS]] uses a stack; [[BFS]] uses a queue.

```python
grid = [[0, 1], [1, 0]]
adj = [[1], [0]]
```

## Related

- [[Heap]]
- [[Missing Page That Does Not Exist]]
