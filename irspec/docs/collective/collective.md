# Collective IR

The goal of this document is to give an overview of the key concepts present in the IR. It does not (yet) fully describe the semantics of the computation.


## Syntax Fundamentals

### Streams
The stream class of the Spatial IR is extended with `multistream<T>`, for a scalar type `<T>`.

If `hops = auto` the routing is optimized while using at most `#channel` channels. For implementation details on the used number of channels see the `Channel Usage` section.

In addition to `hops` and `channel` the operation `op` can be defined for certain collective communication pattern, i.e. reduce. The options for the operation `op` are:

- CL_MAX (returns the maximum element)
- CL_MIN (returns the minimum element)
- CL_SUM (returns the sum of all elements)
- CL_PRODUCT (returns the product of all elements)

### Collective Functions
Collective Communication functions can be called inside the compute block. For further implementation details see the specific collective definition.

## Broadcast
A broadcast is defined with the standard send and receive framework provided by the Spatial IR. It is differentiated from the single point to point communication by using a `multistream` instead of a standard stream.

Sending data in a broadcast that is defined via the multistream `bcast` can therefore be defined as:
```rust
compute i16 variable, i16 variable in subgrid_expression {
  send(a, bcast)
}
```

???+ example "Example: Simple Broadcast"
    ```rust
    compute i16 i, i16 j in [1:N, 0] {
        await receive(a, bcast)
    }

    compute i16 i, i16 j in [0, 0] {
       await receive(a, a_in)
       await send(a, bcast)
    }
    ```
    where `i`, `j` are `i16` variables that are bound to the coordinates of the PEs in the subgrid and `bcast` is a multistream.



## Reduce



## Channel Usage