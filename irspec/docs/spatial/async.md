#  Semantics of Asynchronous Statements

Asynchronous statements may, but do not necessarily run in parallel.
Instead, they may execute in any order and may be interleaved with other statements.
An asynchronous statement may be pre-empted at any time, even partially during its execution.
Hence, it is imperative to properly define their semantics to avoid problems, such as *data races*
and properly define how the representation can be lowered to a task or thread model.


## Local Order

The *local order* defines the 'local' view of the execution of a PE.
It a partial order defined in terms of blocking statements:

!!! abstract "Definition: Blocking Statements"
    A *blocking statement* is a statement that must complete before
    any following statement can start.

In particular:

   - `await` statements are blocking. This includes asynchronous statements that immediately `await` their completion.
   - assignments to fields are blocking.

!!! abstract "Definition: Local Order"
    We say $S_2$ follows $S_1$ in local order and write $S1 \succeq S_2$ if
    $S_1$ and $S_2$ are in the same compute block, $S_2$ follows $S_1$ in all execution paths, and one of the following hold:
    
    - $S_1$ is a blocking statement.
    - $S_1$ is a non-blocking statement with completion `c` and there is a statement `await c` between all execution paths from $S_1$ to $S_2$.

Note that local order does not model loop-carried dependencies,
and instead considers the program order.

??? example "Example: Loops and Local Order"
    ```rust
    // S1
    a[0] = 0;
    for i16 i in [0:10] {
        // S_2
        a[i] = b[i];
        // S_3
        c[i] = a[i] + b[i];
    }
    // S_4
    c[0] = a[0];
    ```
    We have $S_1 \succeq S_2 \succeq S_3 \succeq S_4$ in local order.
    We do not have $S_2$ following $S_3$ in local order because in the last iteration of the loop, $S_2$ is not executed after $S_3$.

## Stream edges

*Stream edges* represent the communication between PEs
and affect the ordering of statements in the `compute` block.

!!! abstract "Definition: Stream Edge"
    A stream edge goes from a statement-PE pair $S_1, (i_1, j_1)$ to a statement-PE pair $S_2, (i_2, j_2)$
    if the data sent from $S_1$ at PE $(i_1, j_1)$ is received by $S_2$ at PE $(i_2, j_2)$.

Our definition of `send` requires that the order in which statements `send`s access a given stream
is in local order. Similarly, the order in which statements `receive` from a stream
is in local order. Moreover, `send`s and `receives` in loops must always line up one-to-one.
Hence, we can always match `send`s and `receive`s in local order to uniquely form stream edges.

## The Happens-Before Relation

The asynchronous semantics can be defined in terms of a *happens-before* relation. 
For each statement $S$ in a `compute` block and each PE $(i, j)$ in the subgrid,
we define a *happens-before* relation $\rightarrow$ between statement-PE pairs.

!!! abstract "Definition: Happens-Before Relation"
    If every instance of $S_1$ at PE $(i_1, j_1$) completes 
    before the next respective instance of $S_2$ starts at PE $(i_2, j_2)$, we say 
    that $S_1, (i_1, j_1)$ happens-before $S_2, (i_2, j_2)$ and write $S_1, (i_1, j_1) \rightarrow S_2, (i_2, j_2)$.

If $S_1, (i_1, j_1) \rightarrow S_2, (i_2, j_2)$ holds for all $(i_1, j_1)$ and $(i_2, j_2)$ in the subgrid, 
we write $S_1 \rightarrow S_2$ for short. This means that the statements are ordered by happens-before
for all PEs in the subgrid. 

!!! note
    The happens-before graph is a formal model used to define the semantics of the language.
    It is not a data structure that is explicitly constructed or used in the implementation.
    Instead, we will see in [Parametric Happens-Before Graph](../parametric) how to efficiently construct
    a compact approximation of the happens-before graph.


We characterize the relation in terms of the local order, `await` statements in the code,
and stream edges. 

!!! abstract "Lemma: Happens-Before Relation"
    We have that $S_1, (i_1, j_1) \rightarrow S_2, (i_2, j_2)$ if *any* of the following hold:

    1. **Local Order**: $S_1 \rightarrow S_2$ are in local order.
    2. **Receive completion implies send completion**:
       $S_1$ is a `send` statement, and $S_2$ is the `await` statement of the corresponding `receive` 
       forming the stream edge from $(S_1, (i_1, j_1))$ to $(S_2, (i_2, j_2))$.
    
    3. **Propagation through stream edges**: 
       There exists a stream edge from some $S_3, (i_1, j_1)$ to $S_4, (i_2, j_2)$ for which:
    
        - $S_1, (i_1, j_1) \rightarrow S_3, (i_1, j_1)$ and 
        - $S_2$ follows $S_4$ on all execution paths.
    
    4. **Transitivity**: There is a $S_3, (i_3, j_3)$ where $S_1, (i_1, j_1) \rightarrow S_3, (i_3, j_3)$ and $S_3, (i_3, j_3) \rightarrow S_2, (i_2, j_2)$.

Note that we handle phases by implicitly adding `await` statements for all outstanding 
completions at the end of each `compute` block.


The happens-before relation is used to define data races.
A compact representation of it can be used for lowering, specifically it can be used to 
determine how the code can be mapped to a task-based or thread-based model.

## Data Races


Two statements are concurrent if they are not ordered by happens-before:
!!! abstract "Definition: Concurrent Statements"
    Two statements $S_1$ and $S_2$ are considered **concurrent** if there exist PEs $(i, j)$ and $(i', j')$
    in the subgrids of $S_1$ and $S_2$ respectively such that
    neither $S_1, (i, j) \rightarrow S_2, (i', j')$ nor $S_2, (i', j') \rightarrow S_1, (i, j)$.

    If $(i, j) = (i', j')$, we say that the statements are *concurrent on the same PE*.

!!! abstract "Definition: Data Race"
    Writing to an array in a statement while concurrently reading from it 
    or writing to it in another statement on the same 
    PE constitutes a *data race* and is considered undefined behavior. 

In particular, sending data from an array while concurrently
writing to it on the same PE is considered a data race.
You must synchronize such statements using `await`.
The motivation for this strict definition is to ensure correctness regardless
of the interleaving of concurrent operations.

???+ example "Example: Data Races"
    ```rust
    // For example, this is a data race:
    // Concurrently writing to and sending from the same array
    // We forbid this because the result of the send would be non-deterministic
    completion c1 = send(a, stream);
    for i16 k in [0:K] {
        // Data Race!!
        a[k] = k; 
    }
    await c1;
    
    // Correctly synchronized, we would get
    completion c1 = send(a, stream);
    await c1;
    for i16 k in [0:K] {
        a[k] = 1;
    }
    
    // Correctly synchronized with short-hand await syntax
    await send(a, stream);
    for i16 k in [0:K] {
        a[k] = 1;
    }
    
    ```

??? example "Example: Ping-Pong"
    Here is an example that synchronized through multiple compute
    blocks using a ping-pong pattern:
    It also includes one statement that demonstrates a data race.
    
    ```rust
    phase {
      // Example: 'Ping-Pong'
      // Ping-pong pattern to synchronize two compute blocks
      // This is a correct way to synchronize two compute blocks
      // that write to the same array
      
      // Send a from 1 to 0
      // at 0, wait for receival, then send to 1
      // at 1, on receival update array a
      
      place i16 i, i16 j in [0:2, 0] {
        f32[K] a;
      }
    
      dataflow i16 i, i16 j in [0:2, 0] {
          stream<f32> eastwards = relative_stream(1, 0);
          stream<f32> westwards = relative_stream(-1, 0);
      }
    
      compute i16 i, i16 j in [0, 0] {
         // S_1
         completion c1 = foreach f32 x, i16 k in receive(eastwards), [0:K] {
            a[k] = 2 * x
         }
         // S_2
         await c1;
         // S_3
         completion c2 = send(a, westwards);
      }
    
      compute i16 i, i16 j in [1, 0] {
         // S_4
         completion c3 = send(a, eastwards);
         // S_5 (data race)
         a[0] = 0;
         // S_6
         completion c4 = foreach f32 x, i16 k in receive(westwards), [0:K] {
            // S_7 (correctly synchronized)
            a[k] = x;
         }
      }
    }
    ```
    Analysis of the Ping-Pong example:

    - We have that $S_4 \rightarrow S_2$ because of the stream edge from $S_4$ to $S_1$ and *receive completion implies send completion*.
    - We have $S_2 \succeq S_3$ in local order.  
    - We have that $S_2 \rightarrow S_7$ because there is a stream edge from $S_3$ to $S_6$
      and all execution paths to $S_7$ go through $S_6$ (*Propagating happens-before through stream edges*).
    - Hence, we have $S_4 \rightarrow S_7$ by transitivity.
      Hence, the statement $S_7$ is correctly synchronized with $S_4$.
    
    However, the access at $S_5$ is concurrent with the `send` at $S_4$.
    This is a data race.


??? example "Example: Sends through the same stream"
    
    Observe that `sends` for a given stream in the same `compute` block are ordered by happens-before
    in the same order as they appear in the code.
    Similarly, for `receives`. However, `sends` and `receive` to the same stream
    can be concurrent or ordered by happens-before in reverse local order.
    
    ```rust
    // Example: Sends to the same stream must be synchronized, and receives as well.
    // They may be concurrent with each other
    
    place i16 i, i16 j in [0:4, 0] {
        f32[K] a;
        f32[K] b;
    }
    
    dataflow i16 i, i16 j in [0:4, 0] {
        stream<f32> eastwards = relative_stream(1, 0);
    }
    
    compute i16 i, i16 j in [0, 0] {
      // Receive twice:
      // The receives must be synchronized
      // S_1
      await foreach i16 k, f32 x in [0:K], receive(eastwards) {
          a[k] = x + 1;
      }
      // S_2
      await foreach i16 k, f32 x in [0:K], receive(eastwards) {
          a[k] = a[k] + x;
      }
    }
    
    compute i16 i, i16 j in [1:4, 0] {
       // S_3
       // Receive (concurrent with send)
       completion c2 = foreach i16 k, f32 x in [0:K], receive(eastwards) {
          // S_4
          a[k] = x + 1;
       }
       // S_5
       await send(a, eastwards);
       
       // S_6
       completion c3 = send(b, eastwards);
       
       // S_7
       await c2;
       
       // S_8
       // Receive (concurrent with send)
       completion c4 = foreach i16 k, f32 x in [0:K], receive(eastwards) {
          // S_9
          a[k] = a[k] + x;
       }
    
       // S_10
       await c3;
       await c4;
    }
    ```
    
    The sends are ordered by happens-before as in the program $S_5 \rightarrow S_6$.
    Similarly, the receives are ordered by happens-before as in the program $S_1 \succeq S_2$ and $S_3 \succeq S_8$.
    However, $S_3$ and $S_5$ are concurrent, as are $S_3$ and $S_6$, as are $S_6$ and $S_8$.


??? example "Example: Ping-Pong-Ping"

    Let's revisit the ping-pong example, but add another ping re-using
    the same stream:
    
    ```rust
    phase {
      // Example: 'Ping-Pong-Ping'
      // Ping-pong-ping pattern
      // that demonstrates implicit synchronization through ping-pong
      
      // Send a from 1 to 0
      // at 0, wait for receival, then send to 1
      // at 1, on receival update array a
      
      place i16 i, i16 j in [0:2, 0] {
        f32[K] a;
      }
    
      dataflow i16 i, i16 j in [0:2, 0] {
          stream<f32> eastwards = relative_stream(1, 0);
          stream<f32> westwards = relative_stream(-1, 0);
      }
    
      compute i16 i, i16 j in [0, 0] {
         // S_1
         completion c1 = foreach i16 k, f32 x in [0:K], receive(eastwards) {
            a[k] = x;
         }
         // S_2
         await c1;
         // S_3
         completion c2 = send(a, westwards);
    
         // Another ping
         // S_4
         await foreach i16 k, f32 x in [0:K], receive(eastwards) {
            a[k] = x;
         }
      }
    
      compute i16 i, i16 j in [1, 0] {
         // S_5
         completion c3 = send(a, eastwards);
         // S_6
         completion c4 = foreach i16 k, f32 x in [0:K], receive(westwards) {
            // S (correctly synchronized)
            a[k] = x;
         }
         // S_7
         await c4;
         
         // Another ping
         // S_8 (implicitly synchronized through the ping-pong)
         completion c5 = send(a, eastwards);
      }
    }
    ```
    In this example, we can argue that:

    - $S_5 \rightarrow S_2$ because of the stream edge from $S_5$ to $S_1$ and *receive completion implies send completion*.
    - $S_2 \succeq S_3$ because $S_2$ is an `await`, which is a blocking statement.
    - $S_3 \rightarrow S_7$ because of the stream edge from $S_3$ to $S_6$ and *receive completion implies send completion*.
    - $S_7 \succeq S_8$ because $S_7$ is an `await`, which is a blocking statement.

    Hence, by transitivity, we have $S_5 \rightarrow S_8$.
    
    Therefore, the sends are correctly synchronized, even though there
    is no explicit `await` on the first send completion.

    The receives are explicitly synchronized.


??? example "Example: 1D Chain Reduce"
    
    So far, we have considered examples with a constant number of PEs.
    In this case, it is not important to differentiate for which PE in the subgrid
    the happens-before relation holds.
    We now consider an example where computation is parameterized,
    which will lead to a more complex happens-before graph,
    whose size depends on the number of PEs and where we need to model
    the PE coordinates explicitly.
    Here is an example that demonstrates a 1D chain reduce with root 0.
    
    
    ```rust
    place i16 i, i16 j in [0:K, 0] {
        f32[K] a;
    }
    
    dataflow i16 i, i16 j in [0:K, 0] {
        stream<f32> eastwards = relative_stream(1, 0);
    }
    
    compute i16 i, i16 j in [0, 0] {
        // S_1
        await foreach x, k in [0:K], receive(eastwards) {
            // S_2
            a[k] = a[k] + x;
        }
    }
    
    compute i16 i, i16 j in [1:K-1, 0] {
        // S_3
        await foreach i16 k, f32 x in [0:K], receive(eastwards) {
            a[k] = a[k] + x;
        }
        // S_4
        completion c1 = send(a, eastwards);
    }
    
    compute i16 i, i16 j in [K, 0] {
        // S_5
        completion c1 = send(a, eastwards);
    }
    ```
    
    Analysis of the Happens-Before Relations:
    
    - $S_4, (1, 0) \rightarrow S_1, (0, 0)$ (by stream edge and *receive completion implies send completion*)
    - $S_4, (i, 0) \rightarrow S_3, (i-1, 0)$ for `i` in `[2:K-1]` (by stream edge and *receive completion implies send completion*)
    - $S_5, (K, 0) \rightarrow S_3, (K-1, 0)$ (by stream edge and *receive completion implies send completion*)
    - $S_3, (i, 0) \succeq S_4, (i, 0)$ for `i` in `[1:K-1]`
    
    Hence, we can conclude by transitivity:
    
    - $S_4, (i, 0) \rightarrow S_4 (i-j, 0)$ for `i` in `[2:K-1]`, `j` in `[1:i-1]`
    - $S_5, (K, 0) \rightarrow S_4, (i, 0)$ for `i` in `[1:K-1]`
    - $S_4, (i, 0) \rightarrow S_1, (0, 0)$ for `i` in `[1:K]`
    - $S_5, (K, 0) \rightarrow S_1, (0, 0)$
    
    The computation is correctly synchronized, and we
    have fully characterized all happens-before relations.

## Deadlocks

Some asynchronous statements cannot make progress until some event occurs.
It is guaranteed that if there exists a statement that
can make progress, at least one of them will make progress.
There is no guarantee of fairness, that is, concurrents statements may be
executed in any respective order and may be preempted at any time.
Failure to guarantee completion regardless of progress order of concurrent operations constitutes a **deadlock**.


In particular, each iteration of a [`foreach`](../spatial#processing-data-streams-with-foreach) stalls until receiving a data element.
An [`await`](../spatial#await-completions-with-await) statement stalls until a completion triggers. 
A [`send`](../spatial#streaming-data-with-send) statement may stall while the receiver is not ready to receive the data.

!!! danger "Deadlock"

    A deadlock-free program ensures that all PEs eventually make progress
    regardless of the interleaving of concurrent statements.

??? example "Example: Deadlock"
    ```rust
    stream s1 = relative_stream(1, 0);
    // ...
    await send(a, s1);
    await foreach i16 k, f32 x in [0:K], receive(s1) {
        a[k] = x;
    }
    ```
    This example constitutes a deadlock.
    The send of s1 cannot make progress until the receive of s1 is complete.
    However, the receive of s1 cannot complete until the send of s1 is complete.

??? example "Example: Deadlock"
    ```rust
    stream s1 = relative_stream(1, 0);
    stream s2 = relative_stream(-1, 0);
    /// ...
    // at P0:
    await send(a, s1);
    await foreach i16 k, f32 x in [0:K], receive(s2) {
        a[k] = x;
    }
    // ...
    // at P1:
    await send(a, s2);
    await foreach i16 k, f32 x in [0:K], receive(s1) {
        a[k] = x;
    }
    ```
    This example constitutes a deadlock.
    The send of s1 cannot make progress until the receive of s2 is complete.
    However, the receive of s2 cannot complete until the send of s2 is complete.
    And the send of s2 cannot make progress until the receive of s1 is complete.


??? example "Example: No Deadlock"
    ```rust
    send(a, s1);
    send(b, s2);
    
    await foreach f32 x in receive(s1) {
        // Process x 
    }
    await foreach f32 x in receive(s2) {
        // Process x
    }
    ```
    This example does not constitute a deadlock.
    Because we guarantee that there is progress if there is some progress to be made.
    In particular, the send of s1 can always make progress.
    Once the receive of s1 is complete, the send of s2 can make progress.
