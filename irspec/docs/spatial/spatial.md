# Spatial IR - High-level IR for spatial computations



## Syntax Fundamentals

### Scalars

`f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `bool` indicate scalar types.

### Constant Literals

We may use constant literals to represent constant compile-time values. 

For example `0`, `1`, `1024`, `-12` are constant literals. 
Constant integer literals are `i64` and constant floating-point literals are `f32`. 


### Streams

A stream corresponds to an abstract way to communicate between PEs or the host device and the PEs.

For any scalar type `T`,  `stream<T>` indicates the corresponding element type sent over the stream.

Streams do not send a predetermined number of elements, but the sender and receiver must agree on the number of elements sent and received.
This can be done explicitly (when the size is known from the parameters) or implicitly (by sending a completion signal with/after the last element).

Kernel arguments that are streams may have a second template parameter `stream<T, K>`. If the second parameter is given, then
exactly `K` elements are transferred over the stream. This is useful for enabling, e.g., memcpy mode in CSL.

### Arrays

Any scalar or stream type `T` and one or more parameter expressions `S_1`, `S_2`, ... `S_d` may be used to create an array type `T[S_1, S_2, ... S_d]`.
It represents a d-dimensional array of type `T`, where the i-th dimension contains `S_i` elements.

For example, `f32[10]`, `i32[I+2, J+2]` indicate array types.

### Parameters

Parameter literals are placeholders for an actual value that will be substituted with an **integer** 
value at compile time. They are denoted by capital letters or capital letters followed by a number string.
For example, `I`, `J`, `K`, `I001` denote parameters.

### Variables

A variable name starts with a lower case letter and may contain letters, numbers, and underscores.
For example, `x`, `y`, `my_variable`, `my_Variable_2` are valid variable names.

A variable declaration contains a type `T` and a variable name.
```
variable ::= [a-z][a-zA-Z0-9_]*
variable_declaration ::= T variable_name
```

A variable is in scope if it is declared in the current block or any enclosing block.

### Parameter Expressions

A parameter expression is an expression that may depend on parameters and constant **integer** literals. 

```
parameter_expression ::= constant_literal | parameter_literal | parameter_expression + parameter_expression | parameter_expression - parameter_expression | parameter_expression * parameter_expression | parameter_expression / parameter_expression | parameter_expression % parameter_expression | (parameter_expression)
```
where / denotes integer division and % denotes modulo.

For example, `I`, `J+2`, `10`, `(I+J) / 2` are parameter expressions.

### Expressions

An expression may depend on parameters, constants, in-scope variables, and fields.

```
array_expression ::= variable[int_expression]
int_expression ::= constant_literal | parameter_literal | variable | expression + expression | expression - expression | expression * expression | expression / expression | expression % expression | (expression)
expression ::= constant_literal | parameter_literal | variable | array_expression | expression + expression | expression - expression | expression * expression | expression / expression | expression % expression | (expression) 
bool_expression ::= expression == expression | expression != expression | expression < expression | expression <= expression | expression > expression | expression >= expression | bool_expression & bool_expression | bool_expression | bool_expression | !bool_expression | (bool_expression)
```
where / denotes integer division and % denotes modulo,
and `==`, `!=`, `<`, `<=`, `>`, `>=`, `&`, `|`, `!` are the standard comparison and logical operators.
Note that the logical operators `&` and `|` do not short circuit. Both operands are always evaluated.

???+ example "Example: Expressions"
    `I`, `J+2`, `i`, `I+i` are integer expressions, `a[k+1]` is an array expression, `I+J < 10` is a boolean expression.

`int` expressions must be of type `i64`. 
`bool` expressions must be of type `bool`. 
`array` expressions must be of type `T[S_1, S_2, ... S_d]`
for some scalar type `T` and parameter expressions `S_1`, `S_2`, ... `S_d`.

in an `expression`, values of type `bool` are automatically converted into integers if needed.
That is, `true` is converted to `1` and `false` is converted to `0`.

??? example "Boolean expressions to emulate conditionals"
    This example computes the minimum of two integers `x` and `y` if they are 
    distinct. If they have the same value, it returns the value `z`.
    ```rust
    b1 = x < y;
    b2 = x == y;
    b3 = x > y;
    result = b1 * x + b3 * y + b2 * z;
    ```
    This demonstrates the casting of boolean expressions to integer values
    and how to implement conditional-like behavior.

### Range expressions

A `range_expression` can be constructed using the following syntax:
```
range_expression ::= start:stop | start | start:stop:step
```
where `start`, `stop`, `step` are integer expressions. If all expressions are parameter expressions, the
range expression is a parameter range expression.
The start is inclusive, and the stop is exclusive. The `step` describes the stride of the range.

### Lists

A list of elements of `X` is separated by commas.

???+ example "Example: Lists"
    For example, `1, 2, 3` is a list of constant literals.
    `x, y, z` is a list of variables.
    `f32[10], i32[I+2, J+2]` is a list of array types.
    `I+2, J+2` is a list of parameter expressions.

### The coordinate grid

The coordinate grid has two dimensions, `x` and `y`.
The origin `(0, 0)` is at the north-west corner of the grid.
The `x` axis increased towards the east, and the `y` axis increases towards the south.


### Subgrid expressions

A subgrid expression is given by 
```
subgrid_expression ::= [parameter_expression, parameter_expression]
```
and it describes a subgrid of the PEs.

For example, `[0:I, 0:J]` describes the entire grid of PEs.
`[0:I:2, 0:J/2]` describes every second PE in the `x` direction and the first half of PEs in the `y` direction.

### Kernel

A kernels abstracts a computation that is executed on a grid of processing elements (PEs).
A kernel is defined using the following syntax:

```rust
kernel kernel_name<parameters> (arguments) {
  // Kernel definition
}
```
where parameters is a list of parameter literals, and arguments is a list of arguments to the kernel.

A kernel gets the memory of its arguments from a host device or other kernel,
runs the computation, and returns the results to the host device.
This is done through communication streams, which are explicitly defined in the kernel arguments.
Inputs and outputs may be sent and received in a streaming fashion.

### Kernel Arguments

An argument is a named and typed stream array or scalar that is passed to a kernel.
```
argument_name ::= [a-z][a-zA-Z0-9_]*
argument ::= T argument_name | T readonly argument_name | T writeonly argument_name | T compiletime argument_name
```
where `T` is a scalar type, stream type, or stream array type.
Notable, it is not possible to pass scalar arrays as arguments, instead,
arrays must be read through streams.

If an argument may be *only* read from or written to, it is marked as `readonly` or `writeonly`, respectively. Stream arguments must be marked with either `readonly` or `writeonly`.

???+ example "Example: Kernel arguments"
    `stream<f32>[I, J] readonly arg1`, `stream<f32>[I, J] writeonly arg2`, `f32 arg3` are arguments.

If an argument is known at compiletime it may be annotated with `compiletime`.
It must be provided at compilation time together with the parameters.


## Place block

All data is placed on the PEs using one or more `place` blocks.

The syntax of a place block is as follows:
```rust
place i32 variable, i32 variable in subgrid_expression {
   // Statements
}
```
For example:
```rust
place i32 i, i32 j in [0:I:2, 0:J] {
    // Statements
}
```
Where `i`, `j` are `i32` variables that are bound to the coordinates of the PEs in the subgrid.

Semantically, a place block iterates over every value in subgrid_expression and 
allocates field memory as described in the block.
The *fields* declared therein become bound to the coordinates of the PEs in the subgrid.
They can be used by statements in a `compute` block.

Field names follow the same syntax as variable names:
```rust
field_name ::= [a-z][a-zA-Z0-9_]*
```

Within a `place` block, the following statements are supported:

- Allocate a local array field: `T[S_0, ...] field_name;`
- Allocate a local scalar field: `T field_name;`


The subgrid of the `place` block is given by the PEs that lie in the `subgrid_expression`.
An array may be placed using multiple `place` blocks.
However, each `field_name` may appear at most once for any given PE over all `place` blocks.

!!! danger "Error: Uniqueness of Field Names"
    Each field name must be unique within a `place` block.

    *Failure to provide unique field names raises a syntax error.*

## Dataflow block

All communication is set up in one or more `dataflow` blocks, which describe the communication streams between PEs.

The syntax of the dataflow block is as follows:
```rust
dataflow i32 variable, i32 variable in subgrid_expression {
  //Statements
}
```
The subgrid of the dataflow block is given by the PEs
that lie in the `subgrid_expression`.
The subgrids of the dataflow blocks must be disjoint.

The dataflow block can be set up to support various types of streams.
Currently, only *relative stream* indexing is supported:

### Relative Stream Declaration

Inside a `dataflow block`, a relative communication stream is declared as follows:
```
stream<T> stream_name = relative_stream(dx, dy);
```
where `T` is a scalar type and `dx` and `dy` are parameter expressions that describe the relative position of the target PE.
This describes a streaming communication stream for sending from the current PE at some
position `(i ,j)` to the PE at the relative position `(i+dx, j+dy)`, and simultaneously
a stream for receiving from the PE at the relative position `(i-dx, j-dy)` at the current PE at `(i, j)`.

??? example "Example: Relative Stream Declaration"
    For example,
    ```rust
    dataflow i16 i, i16 j in [0:I, 0:J] {
        stream<f32> eastwards = relative_stream(1, 0)
        stream<f32> westwards = relative_stream(-1, 0)
        stream<f32> northwards = relative_stream(0, -1)
        stream<f32> southwards = relative_stream(0, 1)
    }
    ```
    describes four communication streams to the east, west, north, and south of each PE.

    For example,
    ```rust
    dataflow i16 i, i16 j in [0:I, 0:J] {
        stream<i32> two_north = relative_stream(0, -2)
    }
    ```
    describes a communication stream that sends `i32` data two PEs to the north. 

!!! note
    The stream declaration does not imply that any data is ever sent over the stream.
    It merely declares the existence of a virtual communication stream.

!!! danger "Error: Uniqueness of Stream Names"
    Each stream name must be unique within a `dataflow` block.

    *Failure to provide unique stream names raises a syntax error.*

### Routing Declarations

Optionally, a routing declaration may be set up for each stream.
This declaration describes how the data is routed between the PEs.
In particular, for each stream, the configuration may specify the intermediate hops that the data takes.
Moreover, it may specify a `channel`, which is a limited hardware resource
(a virtual or hardware channel) that is used to route the data.

The routing configuration is set up as follows:
```
stream<T> stream_name = relative_stream(dx, dy) {
    // Optional routing declaration
    hops = [(dx_1, dy_1), (dx_2, dy_2), ... , (dx_n, dy_n)],
    channel = channel_id
}
```
where `hops` is a list of relative hops that the data takes between the sender and receiver.
Each hop is given by a pair of constant literals, the sum of their absolute value must be 1.
The sum of all the hops must be equal to the relative position of the stream.

If two messages (elements of a `send`) are routed through a PE simultaneously,
it must be ensured that they do not share a `channel`.
Note that the start and end PEs also count as hops implicitly.

???+ example "Example: Routing Declaration"
    ```rust
    dataflow i16 i, i16 j in [0:I, 0:J] {
        stream<f32> eastwards = relative_stream(1, 0) {
            hops = [(1, 0)],
            channel = 0
        }
    }
    ```


If no routing declaration is provided, it is up to the compiler to determine the routing.
This is equivalent to setting `hops = auto` and `channel = auto`.
One may also provide `hops` explicitly, but leave `channel = auto`, which allows the compiler to determine the channel.
```rust
// Example use of channel=auto

dataflow i16 i, i16 j in [0:I, 0:J] {
    stream<f32> eastwards = relative_stream(1, 0) {
        hops = auto;
        channel = auto;
    };
}

```
See the [Semantics of Routing Declarations](../routing) for how the compiler
checks if routing declarations are correct and how it resolves auto-routing.


## Compute block

The computation is described in one or more `compute` blocks.
Computation is inherently asynchronous, triggered by receiving data from streams.
Statements in the `compute` block may return completions that may trigger other statements.

The compute block is defined as follows:
```rust
compute i32 variable, i32 variable in subgrid_expression {
  // Statements
}
```

???+ example "Example: Compute block"
    ```rust
    compute i32 i, i32 j in [0:I, 0:J] {
        // Statements
    }
    ```
    where `i`, `j` are `i32` variables that are bound to the coordinates of the PEs in the subgrid.

The subgrid of the `compute` block is given by the PEs
that lie in the `subgrid_expression`.

`compute` blocks may contain the following statements, some of which are
asynchronous and return completions that may be used to synchronize computations.
```rust
// Send (asynchronous)
completion completion_name = send(local_array, stream_name);

// Foreach loop over a receive() stream until the sender is done (asynchronous)
completion completion_name = foreach variables in receive(stream_name) {
  // Statements
}

// Foreach loop over a receive() stream of defined size (asynchronous)
completion completion_name = foreach variables in [parameter_expressions], receive(stream_name) {
  // Statements
}
// Parallel map (asynchronous)
completion completion_name = map variables in [range_expression] {
  // Assignment statements
}
// Sequential for loop
for variables in [range_expression] {
  // Statements
}
// Asynchronous block (asynchronous)
completion completion_name = async {
  // Statements
}

// Await a completion
await completion_name;
```

An assignment statement is of the form 
```rust
// Assign to an array field
array_expression = expression;
// Assign to a scalar field
field_name = expression;
```

### Asynchronous Execution with Completions

A completion is a built-in identifier type that can be used to control asynchronous execution.

In a well-formed code, every asynchronous element must either have a `completion` definition assigned, or be prefixed
with `await`. This includes asynchronous blocks (`foreach`, `map`, `async`) and asynchronous built-in functions (`send`,
`receive`) that appear in the top-level `compute` scope.

A `completion` object cannot be defined on its own (i.e., `completion c`), nor can it be reassigned. Each completion
name must be unique within a `compute` block.

### Await completions with `await`

Inside a `compute` block, an `await` statement is used to wait for a completion to trigger.
The `await` can be applied to a completion name.
```rust
await completion_name;
```
The `await` can be immediately applied to an asynchronous operation as a shorthand:
```
await operation;
// Is semantically equivalent to:
completion c = operation;
await c;
```

??? example "Example: await"
    ```rust
    // Execute a map and wait for its completion
    await map i32 i in [0:10] {
        // Statements
    }
    // Wait for completion of a send
    await send(local_array, stream_name);
    // Wait for completion of a receive
    await foreach i32 k, f32 x in [0:K], receive(stream_name) {
      // Statements
    }
    // Wait for a completion
    await comp;
    ```

!!! note 
    Note that statements inside an `await` may still be preempted by other asynchronous operations!
    
!!! danger "Undefined Behavior"
    Awaiting the same completion twice is considered undefined behavior.
    

Any completion that is never `await`ed is assumed to have an implicit `await` at the end of its parent `compute` block.

See the [Semantics of Asynchronous Statements](../async#semantics-of-asynchronous-statements) for more details
on the semantics of `await`.

The `awaitall` shorthand is used to await all outstanding completions in a `compute` block.
```rust
completion f = ...
completion g = ...
awaitall;
```
This is equivalent to:
```rust
completion f = ...
completion g = ...
await f;
await g;
```

### Streaming Data with `send`

Inside a `compute` block, the `send` statement sends data asynchronously through a `stream`.

```rust
// Send the whole array
completion completion_name = send(local_array, stream_name);
```
```rust
// Send part of an array
completion completion_name = send(local_array[parameter_range_expression], stream_name);
```
The `local_array` must be allocated for each PE in the subgrid
in some `place` block. Similarly, the `stream_name` must be declared in a `dataflow` block
for each PE in the subgrid.

The `completion_name` is a completion handle that may be used to wait for the completion of the send operation.
Note that the completion is triggered when the data has been sent, not when it is received.
The completion merely indicates that the data in `local_array` may be safely overwritten
without affecting the result of the computation.


!!! danger "Error: Data Races"
    Performing multiple sends to the same stream concurrently is considered a data race on the stream.
    You must synchronize the sends using completions.
    Two sends in the same `compute` block are concurrent if they are not ordered by [`await`](#await-completions-with-await).

As a consequence, within each `compute` block, correctly synchronized `send`s **to the same stream** 
always execute in [local order](../async#local-order).

??? Example "Example: Sends through the same stream"
    For example, the following code correctly synchronizes two sends to the same stream:
    ```rust
    // Send the first half of the array
    completion c1 = send(a[0:K/2], stream_name);
    await c1;
    // Send the rest of the array
    completion c2 = send(a[K/2:K], stream_name);
    ```

### Receiving Streaming Data with `receive`

Inside a `compute` block, the `receive` generator operation wraps a stream to receive a stream of data from it.

```rust
receive(stream_name)
```

Send and receive calls must be compatible with the definitions of the streams in the dataflow blocks
and must be matched across PEs. In particular, if there is a `send` from PE `A` to PE `B`, there must be 
one corresponding `receive` from PE `B` to PE `A`.
Similarly, if there is a `receive` at PE `B`, there must be one corresponding `send` with destination `B`.
Such a pair of matched `send` and `receive`'s for a stream is called a *stream edge* from `A` to `B`.
See the [stream edges](../async/#stream-edges) for more details.
Note that a `receive` operation does not imply that any data is actually received,
it merely declares the existence of a stream edge.

!!! danger "Error: Deadlocks"
    Failure to construct proper stream edges may result in a *deadlock*. The compiler
    will check these constraints and report potential deadlocks on a best-effort basis.

!!! danger "Error: Dataraces"
    Two receives in the same `compute` block are considered concurrent if they are not ordered by `await`.
    Receiving from the same stream multiple times concurrently is considered a data race on the stream.

As a consequence, within each `compute` block, correctly synchronized `receive`s **from the same stream** 
always execute in [local program order](../asnyc/#local-order).


### Processing Data Streams with `foreach`

Inside a `compute` block, a `foreach` loop can be used to apply a computation to a stream of data.
For each element in the stream, the computation is executed.
The elements are processed in the order they are received.

The foreach loop is defined on a generator (i.e., `receive(stream_name)`), and may optionally
accept an additional range iterator (for example, `[0:K]` or `[0:2, 0:N]`). If an additional
range iterator is provided, it is considered as an implicit zip operator, in which the range
will terminate the loop upon completion. This range can be used to provide a fixed number of
elements to receive. Otherwise, the `foreach` loop will receive until the sender is done:

```rust
// Receive until the sender is done
completion completion_name = foreach variables in receive(stream_name) {
  // Assignment statements
}

// Receive a fixed number of elements
completion completion_name = foreach variables in [parameter_range_expressions], receive(stream_name) {
  // Assignment statements
}
```
The variable at the corresponding position to the `receive` generator is bound to the received data.
Its type must match the type of the stream. The other variables are iteration variables.

The order of range iterators and `receive` generator does not matter. A program in its canonical form will place
the `receive` generator last.

For the iteration variables, one may specify multiple parameter range expressions. 
The iteration variables are bound to the indices of the received data, which is
interpreted as a multi-dimensional array in *row-major* order.

If the number of elements received is known, it is preferable to specify it explicitly in order
to allow for performance optimizations.

For example, the following code receives data from `stream_1` for `K` elements
and assigns the received data to the array `a`.
```
completion completion_name = foreach i32 k, f32 x in [0:K], receive(stream_1) {
    a[k] = x;
}
```

The `completion_name` is a completion handle that may be used to wait for the completion of the `foreach` loop.
Note that the completion is triggered when the data has been received, not when it is sent.
After the completion triggers, the stream may be used for other sends or receives.

!!! danger "Error: Nested Asynchronous Statements"
    Every asynchronous statement contained in a `foreach` loop must be 
    [awaited](#await-completions-with-await) inside the loop.
    
    *Failure to await outstanding completions in the loop raises a compilation error.*

!!! danger "Error: Deadlocks"
    The sizes sent and received must match: Each `foreach` loop iterating over a stream that specifies the number of elements to receive,
    must match the number of elements sent over the corresponding [`send`](#streaming-data-with-send) statement.
    
    *Failure to correctly match the sizes sent and received may result in a deadlock and raises a compilation error.*

!!! danger "Error: Nested Send & Receives"
    Every `send` nested in a `foreach` loop must be associated with exactly one `receive` nested in a `foreach`
    or `for` loop in each of the receiving `compute` blocks.

    *Failure to uniquely match nested `send`s and `receives` results
    in incorrect stream edges and raises a compilation error.*

### Receive Statement

A `receive` statement is a shorthand for a `foreach` loop that receives every element in the stream and 
assigns it to an array.

```rust
completion completion_name = receive(stream_name, identifier) {
  // Assignment statements
}
```

Is equivalent to (in case the stream sends K elements):
```rust
completion completion_name = foreach type k, type x in [0:K], receive(stream_name) {
  identifier[k] = x
}
```

### Processing arrays asynchronously with `map`

Inside a `compute` block, the `map` statement is used to apply a computation to each element of an array.
```rust
completion comp = map variables in [parameter_range_expression] {
  // Affine assignment statements
}
```
Every assignment statement must use an *affine expression* of the variables in the range expression.
The motivation for this is to ensure that the map can be efficiently vectorized.

There is no guarantee on the order in which the map is executed.
Hence, the map must not contain loop-carried dependencies.

??? example "Example: Map"
    ```rust
    completion comp = map i32 i, i32 j in [0:10, J] {
        a[i + 2 * j + 1] = i;
    }
    ```

!!! note
    If you need to perform non-affine array accesses, exploit loop-carried dependencies, 
    or nest other asynchronous operations, use a [`for`](#processing-arrays-sequentially-with-for) loop instead.

### Processing arrays sequentially with `for`

Inside a `compute` block, the `for` loop is used to apply a computation to each element of an array in a sequential order.

```rust
for variables in [parameter_range_expression] {
  // Statement
}
```
The range expression must be a parameter range expression.

The number of variables must match the number of dimensions in the parameter range expression.
The variables are bound to the indices given by the range expression.
The variables must be of type `i32`.

A `for` loop does not return completions, it executes sequentially and in-order.

??? example "Example: For Loop"
    For example, a `for` loop can exploit loop-carried dependencies:
    ```rust
    for i32 i, i32 j in [0:I, 0:J] {
        a[i, j] = a[i-1, j] + a[i, j-1];
    }
    ```
    It can also use non-affine indexing:
    ```rust
    for i32 i, i32 j in [0:I, 0:2] {
        a[i*j] = a[i] + a[j];
    }
    ```

!!! danger "Error: Nested Asynchronous Statements"
    Every asynchronous statement contained in a `for` loop must be 
    [awaited](#await-completions-with-await) inside the loop.

    *Failure to await outstanding completions in the loop raises a compilation error.*

!!! danger "Error: Nested Send & Receives"
    Every `send` nested in a `for` loop must be associated with exactly one `receive` nested in a `foreach`
    or `for` loop in each of the receiving `compute` blocks.

    *Failure to uniquely match nested `send`s and `receives` results
    in incorrect stream edges and raises a compilation error.*

### If-else: TODO



### Computing asynchronously with `async`

Inside a `compute` block, an `async` block is used to execute a computation asynchronously.

```rust
completion comp = async {
  // Assignment statements or nested for-loops
}
```


## Phases

One may define multiple phases in a kernel.
Each phase may contain one or more `place`, `dataflow`, and `compute` blocks.

For a `place` block defined in the outermost scope, the fields defined therein are in-scope for all `compute` blocks in all phases.
For a `place` block within a phase, the fields defined therein are in-scope for the `compute` blocks in that phase.

Similarly, for a `dataflow` block defined in the outermost scope, the streams defined therein are in-scope for all `compute` blocks in all phases.
For a `dataflow` block within a phase, the streams defined therein are in-scope for the `compute` blocks in that phase.

Within each phase, there can be at most one `compute` block defined per PE.
If multiple `compute` blocks are defined per PE per phase, the behavior is undefined.
After each `compute` block, there is a set of implicit `await` statements
that wait for all completions to be triggered before starting the next `compute` block.
Note that this does *not* imply that all PEs have executed the `compute` block.
No `compute` block may be defined in the outermost scope.

Phases run in the order they are defined in the code from each PE's point of view.
That is, a PE goes through its phases in-order.
A PEs may participate in some phases and not in others.

??? example "Example: Phases"
    ```rust
    place i16 i, i16 j in [0:I, 0:J] {
        f32[K] a;
    }
    
    dataflow i16 i, i16 j in [0:I, 0:J] {
      stream<f32> input = arg1[i, j, 0:K];
    }
    
    phase {
      place i16 i, i16 j in [0:I, 0:J] {
        f32[K] b;
      }
       
      dataflow i16 i, i16 j in [0:I, 0:J] {
        stream<f32> eastwards = relative_stream(1, 0);
      }
      
      compute i16 i, i16 j in [0:I, 0:J] {
         // Within this compute block:
         // b and a are in scope, eastwards is in scope, input are in scope
      }
    
    }
    
    phase {
    
      place i16 i, i16 j in [1:I-1, 1:J-1] {
        f32[K] c;
        stream<f32> output = arg2[i, j];
      }
    
      dataflow i16 i, i16 j in [1:I-1, 1:J-1] {
        // The communication pattern switches direction in this phase
        stream<f32> westwards = relative_stream(-1, 0);
      }
      
      compute i16 i, i16 j in [1:I-1, 1:J-1] {
        // Within this compute block:
        // c is in scope, westwards, input and output are in scope
      }
    
    }
    ```
