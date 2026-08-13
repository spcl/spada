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

!!! danger "Error: Concurrent Channel Use"
    If two paths $P_1$ and $P_2$ in the routing graph use the same channel, share a PE, and
    their corresponding stream edges are not ordered by empties-before, then the behavior is undefined.
    *This raises a compile error whenever the missing ordering can be established statically.*


This is because the two messages may interfere with each other
and the order in which they are processed may become nondeterministic.
Recall that sending onto the same stream [must be synchronized using completions
to avoid data races](../spatial#streaming-data-with-send). Hence, sending through the same stream multiple times
in the same phases is ok as long as the sends (and receives) are correctly synchronized.

The constructive way to establish the ordering between two streams that share a channel is to
[close](../spatial#closing-streams-with-close) the earlier one. Closing a stream ends its *epoch*:
on every PE of the path, the channel is released and may be taken over by the next stream.

!!! abstract "Definition: Channel Epoch"
    An *epoch* of a channel $C$ at PE $(i, j)$ is a maximal interval during which a single stream
    that uses $C$ occupies $(i, j)$. It begins at the first use of that stream and ends at its
    `close` (which, for a [bounded](../spatial#streams) stream, is implicit after its `BOUND`
    elements have been transferred, and, for any stream, is implicit at the end of its phase).

!!! abstract "Lemma: Sufficient Condition for Channel Reuse"
    Let $F_1$ and $F_2$ be two streams that use the same channel $C$, and let $(i, j)$ be a PE
    shared by their paths. Let $S_c$ be the `close` of $F_1$ at $(i, j)$ and $S_u$ the first use of
    $F_2$ at $(i, j)$. If $S_c, (i, j) \longmapsto S_u, (i, j)$ for every shared PE $(i, j)$,
    then the stream edges of $F_1$ empty-before those of $F_2$ and the reuse of $C$ is well-defined.

Note that a phase boundary satisfies the condition of the lemma at every PE, which is why streams
in different phases may share a channel without an explicit `close`.

Within a single epoch, a stream may not be used in two different route configurations at the same
PE. In particular, a PE that both receives from and sends on the same channel must close the
channel in between, since the two uses require incompatible router configurations.

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


## Lowering to Switches

Channels are a scarce resource: each channel that is live at a PE occupies one of the hardware's
routing colors. Epochs are what makes it possible to reuse a channel, and hence a color, for
several streams.

A stream induces, at each PE of its path, a *route configuration*: the set of directions the PE
receives from and the set of directions it transmits to (where `RAMP` denotes the PE's own compute
element). Consider a fixed channel $C$ and a fixed PE $(i, j)$. Ordering the streams that use $C$
at $(i, j)$ by their epochs yields a sequence of route configurations
$R_0, R_1, \dotsc, R_{n-1}$, which is realized by the PE's *switch* for the color assigned to $C$:
$R_0$ is the initial configuration and the router *advances* to $R_{k+1}$ at the epoch boundary.

!!! danger "Error: Too Many Route Configurations"
    A router holds a bounded number of route configurations per color (four on both WSE-2 and
    WSE-3). *If the streams sharing a channel require more configurations than that at a single PE,
    a compile error is raised.* Assigning a different channel to some of the streams resolves it,
    at the cost of an additional color.

Consecutive configurations that are equal do not consume a position and do not require an advance.
This is a common case: two streams declared as `relative_stream(-2, 0)` in successive phases induce
the same configuration at every PE of their paths, so their shared channel needs no switching at
all.

An advance is driven by the `close` that ends the epoch, and is emitted only at those PEs whose
next configuration differs. Two lowerings are available:

- The sending PE marks the last transfer of a bounded stream so that its router advances once the
  stream's `BOUND` elements have left the fabric. This adds no traffic to the channel.
- The sending PE emits a *switch-advance control message* on the channel. It follows the stream's
  path using the configuration that is being retired, and advances the router of each PE it
  traverses, after all data of the epoch. PEs on the path whose configuration does not change are
  skipped.

Because the control message travels the path of the retired configuration in order behind the data,
a receiving PE needs to emit nothing to advance its own router: the ordering required by the
[lemma above](#undefined-behavior) is provided by the fabric. A receiver's `close` therefore has no
runtime effect; it exists so that the lifetime of the stream — and hence the number of elements it
carries — is stated by every participant and can be checked.

!!! note "Note: Number of Control Messages"
    A single control message can carry advance commands for a bounded number of consecutive routers
    (eight on both WSE-2 and WSE-3). *A path that would require more raises a compile error.*
