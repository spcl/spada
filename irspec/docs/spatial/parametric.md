# Parametric Semantic Representations

So far, we have introduced and discussed the formal definitions of [stream edges](../async#stream-edges),
the [happens-before graph](../async#the-happens-before-graph), and the [routing graph](../routing/#the-routing-graph).
Next, we discuss how to construct compact, *parametric* representations of these graphs.
The advantage of these representations is that there size is much smaller
than the size of the program grid, and constructing them is polynomial time in 
the size of the program.

## Constructing (Strict) Local Order

### Local Order

The program can be extracted from the control [flow graph](https://en.wikipedia.org/wiki/Control-flow_graph)
(CFG) of basic blocks.
Then, compute the [dominators](https://en.wikipedia.org/wiki/Dominator_(graph_theory)) and post-dominators.

If $S_1$ and $S_2$ are in the same basic block, then $S_1 \succeq S_2$ if

- $S_1$ is blocking and $S_2$ follows $S_1$ in the basic block, or
- $S_1$ is non-blocking and there is an `await` on the completion of $S_1$ between $S_1$ and $S_2$.

Otherwise, $S_1 \succeq S_2$ if:

- $S_1$ is blocking and $S_2$ post-dominates $S_1$, or
- $S_1$ is non-blocking and the set of dominators of $S_2$ contains an `await` on the completion of $S_1$.

#### Analysis

Explicitly constructing the local order costs $O(n^2)$ time, where $n$ is the number of statements in a `compute` block.
However, we can represent the local order implicitly by storing the post-dominator-tree.
Then, we can check if $S_1 \succeq S_2$ in $O(d)$ time, where $d$ is the diameter of the CFG
by a search in the dominator or post-dominator tree if they are in different blocks and comparing
their indices if they are in the same block.

!!! note
    Efficient and practical algorithms can compute 
    [dominator trees in near-linear time](https://www.researchgate.net/publication/220639563_Finding_Dominators_in_Practice).


### Strict Local Order

We can compute the strict local order as a subset of the local order.
For each block $B_2$, determine the set of reachable blocks. 
For each such block $B_1$ reachable from $B_2$ exclude all
$S_1 \succeq S_2$ from the strict local order for all statements $S_1$ in $B_1$ and $S_2$ in $B_2$.

#### Analysis

The implementation of the reachability checks
takes $O(b^2 s^2)$ time, where $b$ is the number of basic blocks
in the control flow graph and $s$ is the largest number of statements in a block.

## *Parametric* Stream Edges

The parametric stream edges are a compact representation of the stream edges.
Each stream edge is represented as $(S_1, (i, j)) , (S_2, (i+d_x, j+d_y)) \ | \ P_1$ for statements
$S_1$, $S_2$, predicated by $P_1$.
Here, $i$ and $j$ are variables that appear in the predicate.
The interpretation is that if the predicate $P_1$ is true for some $i$, $j$ where
$(i, j)$ is in the compute block of $S_1$, then the stream edge exists for the PE $(i, j)$.

The first step to constructing stream edges is to determine the
order of the `send`s and `receive`s that occur to the same stream within each
`compute` block. This follows immediately from the local order $\succeq$.

*The following assumes that `compute` blocks and `dataflow` blocks
match N-1, that is, each compute block is specified by a single dataflow block
from its phase and a single global dataflow block.
We can remove this assumption by splitting the compute block into multiple blocks
until the condition is satisfied.*

Next, we consider each `compute` block in a phase.
We rename all variables in the `compute` and `dataflow` subgrid expressions to use $i$ and $j$ for simplicity.
A `compute` block is now identified with some set of PEs defined as `i, j in [I_1:I_2:I_3, J_1:J_2:J_3]`.
Within this block, consider some `send` statement $S_1$, which is the k-th `send` statement to its stream in the local order.
Let $(d_x, d_y)$ be the offset of its stream. That is, a PE $(i, j)$ sends to PE $(i+d_x, j+d_y)$.
Note that we assume the strides are positive without loss of generality.

We now construct the stream edges for this `send` statement.
For this, we observe the following constraints on the blocks $[I_4:I_5:I_6, J_4:J_5:J_6]$ that can receive from the stream
when sending from PE $(i, j)$:

Range constraints:

- $I_4 \leq i + d_x < I_5$
- $J_4 \leq j + d_y < J_5$

Congruence constraints:

- $i = I_4 - d_x \mod I_6$  in case $I_6 > 1$
- $j = J_4 - d_y \mod J_6$  in case $J_6 > 1$

To correctly identify the stream edges, we need to check if there exists an $(i, j)$ in the current
`compute` block for which all constraints are satisfied.

For this, first solve the linear congruence relations for $i$ and $j$ (in case $I_6 > 1$ and $J_6 > 1$, respectively).
We can efficiently characterize the set of solution (and determine if it is empty).

??? info "Algorithm: How to Solve the Congruence Relations"
    We use that $i=I_1+ x \cdot I_3$ and $j=J_1+ y c \cdot J_3$ for some $x$ and $y$.
    Then, we need to solve for $x$ and $y$ in the following equations:

    - $x \cdot I_3 = (I_4 - I_1 - d_x) \mod I_6$
    - $y \cdot J_3 = (J_4 - J_1 - d_y) \mod J_6$

    Let's focus on the first equation, as the second is symmetric.
    
    1. Compute $\text{gcd}(I_3, I_6) = d$ using the [Euclidean Algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm).
    2. Check if $d$ divides $I_4 - I_1 - d_x$. If not, no solution exists (there is no stream edge).
    3. If $d$ divides $I_4 - I_1 - d_x$, we can solve the equation:
    
        - Simplify the equation by dividing everything by $d$.
        - Solve the simplified equation using the
          [Extended Euclidean Algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm) to find one solution $x_0$:
        - The general solution is: $x = x_0 + l \frac{I_6}{d}$ for $l \in 0, 1, \dotsc , d-1$
        

We filter out all solutions $x$ for which $I_1 + x \cdot I_3 \geq I_2$ as they are out of bounds of the `compute` block.

Now, we can apply the range constraints using the general solution of $x$
to determine the solutions $I_1 + x \cdot I_3 + d_x$ that are in the receiving `compute` block.

That is, we check if 
$I_4 \leq I_1 + x \cdot I_3 + d_x < I_5$ for some $x$ in the general filtered solution of $x$
and similarly for $y$.

If all constraints are satisfied, we have found a valid receiving block.
We match the send statement $S_1$ with the $k$-th `receive` statement $S_2$ in the local order
of the receiving block. If no such `receive$ statement exists, we have a deadlock.

Next, we construct the predicate $P_1$ that describes which PEs in the sending
block send to the receiving block, which is given by the range and congruence constraints.
We may simplify the range constraints, as one of the two inequalities is trivially true
depending on if $d_x$ is positive or negative (and similarly for $d_y$). Moreover, the congruence constraints
may be left out if $I_6 = 1$ or $J_6 = 1$.

We add parametric stream edge $S_1, (i, j)$ to $S_2, (i+d_x, j+d_y)$ predicated by $P_1$
to the list of stream edges.

!!! info "Note: Checking for Correct Stream Edges"
    The algorithm also checks for certain deadlocks by ensuring that all `send`s are matched with a `receive`.
    To check if all `receive`s are matched with a `send`, we can use a similar algorithm
    with the roles of `send` and `receive` reversed and using $d_x$ and $d_y$ negated.
    Once we have the stream edges, we can check if the sizes of the stream edges are consistent
    between sends and receives.

### Analysis

The algorithm takes $O(n^2)$ time overall, where $n$ is the number of `compute` blocks.
One can speed up the algorithm by filtering out all `compute` blocks that are not in the range of the stream
before solving the congruence relations. This can be done efficiently using 2D box intersection tests.

## *Parametric* Happens-Before Graph

We now define a parametric happens-before multi-graph that describes the happens-before relations compactly.
The vertices in the graph are statements and the edges are annotated with $(d_x, d_y)$ constant offsets 
and a predicate over variables $i$ and $j$.
This predicate may also involve parameters of the kernel and constants.
We allow for range constraints and congruence constraints as in the parametric stream edges.
The meaning of the edge from $S_1$ to $S_2$ with offset $(d_x, d_y)$ and predicate $P_1$ is that
if the predicate $P_1$ is true for some $i_1$, $j_1$ in the compute block of $S_1$, 
then $S_1, (i_1, j_1) \rightarrow S_2, (i_1+d_x, j_1+d_y)$. 
If such an edge exists, we write $S_1, (i, j) \rightarrow S_2, (i+d_x, j+dy) \ | \ P_1$.

We do not explicitly represent the transitively closed relation, but only the necessary edges.
One can then follow paths in the graph to determine if two statements are in the happens-before relation
for a given set of PEs.

???+ example "Example: Parametric Happens-Before"
    For example, $S_1$ and $S_2$ $S_3$ could be vertices in the graph.
    Then, we might have $S_1, (i, j) \rightarrow S_2, (i+1, j) \ | i < I-1$
    and $S_1, (i, j) \rightarrow S_3, (i+1, j+1) \ | i = I-1$.


We can construct the parametric happens-before graph for each phase with the following steps.
Its construction follows the same rules as the happens-before lemma, except
we do not explicitly resolve the transitivity to save space and time.

!!! info "Algorithm: Constructing the Parametric Happens-Before Graph"
    Initialize the vertices with the statements in the phase.
    Throughout, if an edge already exists with the same predicate and offset, we do not add it again.
    
    1. For each pair of statements $S_1$, $S_2$ in the same compute block for which 
    $S_1 \succeq S_2$ in [local order](../async#constructing-the-local-order):
    
        - Add the edge $S_1, (i, j) \rightarrow S_2, (i, j)) \ | \ \emptyset$ (true).

    2. For each parametric stream edge from $S_1, (i, j)$ to $S_2, (i+d_x, j+d_y)$ predicated with $P_1$:

        - Add the edge $S_1, (i, j) , S_2, (i+d_x, j+d_y) \ | \ P_1$.
    
    3. For each parametric stream edge from $(S_3, (i, j))$ to $(S_4, (i+d_x, j+d_y))$ predicated by $P_1$:
    
        - Consider all edges $S_1, (i, j) \rightarrow S_3, (i, j) \ | \ P_2$
        and all vertices $S_2$ for which $S_2$ follows $S_4$ in the CFG.
        - Add an edge $S_1, (i, j) \rightarrow S_2, (i+d_x, j+d_y) \ | \ P_1 \land P_2$.

    Whenever creating a new predicate from two predicates, we simplify the predicate
    as much as possible. The resulting predicate remains a conjunction of range constraints
    and congruence constraints.
    To simplify the congruence constraints, we can use the
    [Chinese Remainder Theorem](https://en.wikipedia.org/wiki/Chinese_remainder_theorem).
    If a predicate becomes unsatisfiable after simplification, we skip the edge.


### Strict Happens-Before

To construct the parametric strict happens-before graph, we modify the algorithm as follows:

1. Use the strict local order.

2. Only consider stream edges outside of loops.

3. Only consider vertices $S_2$ for which $S_4 \succ S_2$ in strict local order.

### Analysis

The runtime is polynomial in the number of compute blocks and the number of stream edges
in a phase. [TODO: exact cost analysis]


## *Parametric* Routing Graph

As the routing graph has a size that grows with the size of the PE grid,
we will not construct it explicitly.
Instead, we describe a *parametric* routing graph that describes the routing
graph in terms of predicated edges.

We again consider a particular phase.
The parametric routing graph of a phase is defined as follows:

There is a node for each compute block in the phase.
We rename all variables in the `compute` and `dataflow` subgrid expressions to $i$ and $j$ for simplicity.
The node is identified with the `i, j in subgrid_expression` that describes the PE coordinates of the compute block
and the two variables are bound to the PE coordinates in the compute block.
For example, $[0:I:2, 0:J:1]$ could be a node in the routing graph.

The edges of the parametric routing graph are defined as follows:

For each stream $F$, go over all `send` statements $S_1$ in the local order.
Let $F$ have `hops` = $[(d_{x_1}, d_{y_1}), \dotsc , (d_{x_h}, d_{y_h})]$
and let $v$ be the compute block of $S_1$.

We iteratively add edges as follows, the idea is to explore an implicitly defined
graph using DFS:
Initialize a stack of vertex-index pairs to visit and a set of visited vertex-index pairs.
Add the node-index pair $(v, 1)$ to the stack.

Until the stack is empty:
Pop the top vertex-index pair $(u, k)$ from the stack.
Consider the current vertex $u=[I_1:I_2:I_3, J_1:J_2:J_3]$ and the next hop $(d_{x_k}, d_{y_k})$ at index $k$.
Add an edge $(u, w)$ to each of the vertices $w$ described hereafter,
labeling it with $(F, v, k)$.
If $k < h$ and $(w, k+1)$ is not in the visited set, add $(w, k+1)$ to the stack.

**Case: The stride is $I_3 = J_3 = 1$:**

- To block $[I_1:I_2:1, J_1:J_2:1]$ with predicate 
(the cases are mutually exclusive by definition because $|d_{x_k}|+|d_{y_k}|=1$):

    - $i + 1 < I_1 - 1$ if $d_{x_k} = 1$
    - $i - 1 > I_1$ if $d_{x_k} = -1$
    - $j + 1 < J_1 - 1$ if $d_{y_k} = 1$
    - $j - 1 > J_1$ if $d_{y_k} = -1$

If no such block exists, this constitutes an incorrect declaration of stream edges (deadlock).

- If $d_{x_k} \neq 0$, to all blocks $[I_4:I_5:1, J_4:J_5:1]$ for which $J_4 < J_2$, $J_5 \geq J_1$, and

    - for which $I_4 = I_2 + 1$ with the predicate $i = I_2 \land J_4 \leq j < J_6$ if $d_{x_k} = 1$
    - for which $I_5 = I_1 - 1$ with the predicate $i = I_1 \land J_4 \leq j < J_6$ if $d_{x_k} = -1$
    - Note that the ranges $J_4:J_5$ of all such blocks must together cover the range $J_1:J_2$.


- Proceed symmetrically in case $d_{y_k} \neq 0$.


**Case: All compute blocks have the same stride > 1:**

- If $d_{x_k} \neq 0$, to all blocks $[I_4:I_5:I_3, J_4:J_5:J_3]$ for which $J_4 < J_2$, $J_5 \geq J_1$, and 
for which $I_4 = I_2 + d_{x_k} \mod I_3$ with the predicate $I_4 \leq i + 1 < I_5 \land J_4 \leq j < J_6$. 
    Note that the ranges $J_4:J_5$ of all such blocks must together cover the range $J_1:J_2$, and similarly, 
    the ranges $I_4:I_5$ must together cover the range $I_1+d_{x_k}:I_2+d_{x_k}$.
    Failure to do so constitutes an incorrect declaration of stream edges (deadlock).


- Proceed symmetrically in case $d_{y_k} \neq 0$.



**General Case**

In the general case, we can use the same constraints & method we used to compute stream edges
for the current hop $(d_{x_k}, d_{y_k})$, it uses congruence relations to determine the receiving blocks.

*Note, I did the special cases first, so I kept them for now.
We can also use the general algorithm for all cases,
but we should make sure to be able to simplify all the predicates in the special cases.*

### Analysis

Note that blocks with a single element can be interpreted as having any arbitrary stride.
This is useful for implementing boundary conditions.

Runtime: Note that each vertex is added to the stack at most $h$ times, where $h$ is the number of hops.
Adding all edges for a given vertex takes at most $n$ time,
where $n$ is the number of `compute` blocks.
Hence, the overall runtime is $O(n^2 \cdot h)$.
The space complexity is $O(n \cdot h)$.

## Parametric Strict Happens-Before

We represent the strict happens-before relation parametrically as $S_1, (i, j) \longmapsto S_2, (i+d_x, j+d_y) | P_1(i, j)$, 
which means that if $P_1(i, j)$ is true for some $i, j$ in the compute block of $S_1$, then $S_1 (i, j) \longmapsto S_2 (i+d_x, j+d_y)$.

The parametric strict happens-before relation is constructed by repeatedly applying the constructive
definition of the strict happens-before relation (see [routing](../routing)) using the parametric
stream edges. Whenever the relation is constructed from a stream edge, we add the predicate of the stream edge
to the predicate of the strict happens-before relation. We do not apply the transitive closure explicitly.

## Parametric Empties-Before

We represent the empties before relation parametrically as $e_1 [i, j] \longmapsto e_2 [i + d_x, j + d_y] \mid P_1(i, j)$,
meaning that if $P_1(i, j)$ is true for some $i, j$ in the compute block of source of $e_1$, then $e_1$ empties before $e_2$, where
the coordinates of $e_2$ have been shifted by $(d_x, d_y)$.

The parametric empties-before relation is constructed by checking every pair of parametric stream edges
$(S_1, (i, j) \rightarrow S_2, (i+d_x, j+d_y) \mid P_2)$ 
and 
$(S_3, (i, j) \rightarrow S_4, (i+d_x, j+d_y) \mid P_3)$ 

if $S_2, (i, j) \longmapsto S_3, (i+d_x', j+d_y') \mid P_4$ exists,
adjusting the coordinates of the involved predicates and edges, creating a new predicate $P_1$ that 
expressed the conditions adequately [TODO define exactly how].


## The Conflict Graph

The conflict graph can be used to determine if a routing declaration is correct,
and resolve the `auto` routing declarations.
The conflict graph is a directed graph that describes the conflicts between streams.
Two streams conflict if they are routed through the same channel at some shared PE
and are not ordered by happens-before.

We use the parametric routing graph to construct the conflict graph.



