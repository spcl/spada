# Semantics of Routing Declarations

Routing declarations must respect the limitations on how `channel`s are used.
Specifically, it must be avoided that two messages are routed through the same `channel` 
at the same PE simultaneously.


## Strict Local Order

The *strict local order* strengthens the local order.

!!! abstract "Definition: Strict Local Order"
    We say $S_2$ follows $S_1$ in strict local order and write $S_1 \succ S_2$ if $S_1 \leadsto S_2$
    and additionally, $S_1$ does not follow $S_2$ in any execution path.

??? example "Example: For-Loops and Strict Local Order"
    This differs from the local order in the case of loops. For example:
    ```rust
    // S_1
    a[0] = 0;
    for i in [0:10] {
        // S_2
        a[i] = b[i];
        // S_3
        c[i] = a[i] + 1;
    }
    ```
    We have $S_1 \succ S_2$ and $S_1 \succ S_3$.
    However, $S_3$ does not follow $S_2$ in strict local 
    order because it is in a loop, so sometimes $S_2$ follows $S_3$.


## Strict Happens-Before Relation

We define a strict happens-before relation on statement-PE pairs that stengthens the happens-before relation.
It is used to define the semantics of routing declarations
and for the lowering of routing declarations.

!!! abstract "Definition: Strict Happens-Before Relation"
    If all instances of a statement $S_1$ complete at PE $(i_1, j_1)$ 
    before the first instance of a statement $S_2$ starts at PE $(i_2, j_2)$, 
    then we say that $S_1$ *strictly happens-before* $S_2$ and write $S_1, (i_1, j_1) \longmapsto S_2, (i_2, j_2)$.

We may characterize the relation constructively, similar to the happens-before relation:

!!! abstract "Lemma: Strict Happens-Before Relation"
    We have that $S_1, (i_1, j_1) \longmapsto S_2, (i_2, j_2)$ if *any* of the following hold:

    1. **Strict Local Order**: $S_1 \succ S_2$ are in strict local order.
    2. **Receive completion implies send completion**:
       $S_1$ is a `send` statement **outside of a loop**, and $S_2$ is the `await` statement of the corresponding `receive` 
       forming the stream edge from $(S_1, (i_1, j_1))$ to $(S_2, (i_2, j_2))$.
    
    3. **Propagation through stream edges**: 
       There exists a stream edge from some $S_3, (i_1, j_1)$ to $S_4, (i_2, j_2)$ for which:
    
        - $S_1, (i_1, j_1) \longmapsto S_3, (i_1, j_1)$ and 
        - $S_4 \succ S_2$ in strict local order.
    
    4. **Transitivity**: There is a $S_3, (i_3, j_3)$ where $S_1, (i_1, j_1) \longmapsto S_3, (i_3, j_3)$ and $S_3, (i_3, j_3) \longmapsto S_2, (i_2, j_2)$.


## The Routing Graph

The routing graph of a phase is a directed graph that describes how data is routed between PEs.
Note that the routing graph is defined in terms of the PE coordinates, so
its size grows with the size of the PE grid. It serves as a formal model for defining
the semantics, but should not be constructed explicitly.

Recall that stream edges are pairs of [send](../spatial#streaming-data-with-send) 
and [receive](../spatial#receiving-streaming-data-with-receive) operations that are matched across PEs.
Stream edges must not cross [phases](../spatial#phases), that is, a stream edge must be entirely contained within a phase.

The routing graph contains the following nodes $V$, edges $E$, and paths $P$:

- Each PE is a node in the graph.
- Consider each stream edge from PE $(x_1, y_1)$ to PE $(x_2, x_2)$ going through stream $F$ on channel $C$ through PE `hops` $[(dx_1, dy_1), (dx_2, dy_2), \dotsc , (dx_n, dy_n)]$.
We add an edge from $(x_1+dx_i, y_1+dy_i)$ to $(x_1+dx_{i+1}, y_1+dy_{i+1})$ for each $i$ in $0, \dotsc, n$.
where we use the convention that $dx_0 = dy_0 = 0$.
- Moreover, we add the resulting path $(x_1, y_1), \dotsc, (x_1+dx_i, y_1+dy_i), ..., (x_2, y_2)$ to the list of paths $P$
and record the stream $F$, channel $C$, and corresponding stream edge.

??? example "Example: 2-phase Reduce"
    For example, the following code correctly sets up
    a routing declaration for a 1D 2-phase reduce for 4 PEs:
    It can use a single channel for both phases, as the streams
    are properly sequenced in different phases.
    ```rust
    // 1D 2-phase reduce for 4 PEs
    place i16 i, i16 j in [0:4, 0] {
        f32[K] a
    }
    
    phase {
      dataflow i32 i, i32 j in [0:4, 0] {
        stream<f32> hop1 = relative_stream(-1, 0) {
          hops = [(-1, 0)],
          channel = 0
        };
      }
      compute i32 i, i32 j in [1:4:2, 0] {
        await send(a, hop1)
      }
      compute i32 i, i32 j in [0:4:2, 0] {
        await foreach f32 k, i32 x in [0:K], receive(hop1) {
          a[k] = a[k] + x
        }
      }
    }

    phase {
      dataflow i32 i, i32 j in [0:4, 0] {
        stream<f32> hop2 = relative_stream(-2, 0) {
          hops = [(-1, 0), (-1, 0)],
          channel = 0
        }
      }

      compute i32 i, i32 j in [2, 0] {
        await send(a, hop2)
      }

      compute i32 i, i32 j in [0, 0] {
        await foreach f32 k, i32 x in [0:K], receive(hop2) {
          a[k] = a[k] + x
        }
      }

    }
    ```
    
    The routing graphs for this example contains 4 nodes, one for each PE.
    In the routing graph for the first phase,
    there are two edges from PE (1, 0) to PE (0, 0) and PE (3, 0) to PE (2, 0).
    In the second phase,
    There is a single edge from PE (2, 0) to PE (0, 0).

## Undefined Behavior

Next, we describe the condition under which the routing behavior is undefined:

!!! abstract "Definition: Empties-Before of Stream Edges"
    We say that a stream edge $(S_1, (i_1, j_1), S_2, (i_2, j_2))$
    empties-before stream edge $(S_3, (i_3, j_3), S_4, (i_4, j_4))$
    and write $(S_1, (i_1, j_1), S_2, (i_2, j_2)) \mapsto (S_3, (i_3, j_3), S_4, (i_4, j_4))$
    if $S_2, (i_2, j_2) \longmapsto S_3, (i_3, j_3)$.

!!! danger "Error: Undefined Behavior"
    If two paths $P_1$ and $P_1$ in the routing graph use the same channel, share a PE, and
    their corresponding stream edges are not ordered by empties-before, then the behavior is undefined.


This is because the two messages may interfere with each other
and the order in which they are processed may become nondeterministic.
Recall that sending onto the same stream [must be synchronized using completions
to avoid data races](../spatial#streaming-data-with-send). Hence, sending through the same stream multiple times
in the same phases is ok as long as the sends (and receives) are correctly synchronized.

Keep in mind that PEs transition between phases asynchronously,
that is, a PE may advance to the next phase before another PE has completed the current phase.
We exploit here implicitly that routers back-pressure when
they receive data from a channel on which they are not configured
to receive. 

!!! note "Note: Correctness Conditions"
    The correctness conditions are tailored to the case
    where all streams are point-to-point paths.
    If multicasting is used, the correctness conditions must be adapted accordingly, 
    especially when considering multiple phases.
