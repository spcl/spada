# Collective IR

The goal of this document is to give an overview of the key concepts present in the IR. It does not (yet) fully describe the semantics of the computation.


## Syntax Fundamentals

### Streams
The stream class of the Spatial IR is extended with `multistream<T>`, for a scalar type `<T>`.

MultiStreams take a name and a root in (x,y) coordinates as arguments. Additionally a Broadcast or Reduce can be defined.

### Collective Functions
Collective Communication functions can be called inside the compute block. For further implementation details see the specific collective definition.

## Broadcast
A broadcast is defined with the standard send and receive framework provided by the Spatial IR. It is differentiated from the single point to point communication by using a `multistream` instead of a standard stream. This mimics the support for broadcast communication found in many spatial architectures.

In the dataflow block a broadcast is defined in the following way:
```
multistream<f32> name = broadcast_stream(root_x, root_y) {
      channels = auto
  }
```
where (root_x, root_y) defines the sender. The name is important to give as an argument in the compute blocks. With channels a specific channel can be targeted for the communication in architectures that support it. In almost all situations auto should lead to optimal results.

Sending data in a broadcast that is defined via the multistream `bcast` can therefore be defined as:
```rust
compute i16 variable, i16 variable in subgrid_expression {
  send(data, bcast)
}
```
where data is the data being sent.

???+ example "Example: Simple Broadcast"
    ```rust
    compute i16 i, i16 j in [1:N, 0] {
        await receive(a, bcast)
    }

    compute i16 i, i16 j in [0, 0] {
       await send(a, bcast)
    }
    ```
    where `i`, `j` are `i16` variables that are bound to the coordinates of the PEs in the subgrid and `bcast` is a multistream.

    This can be generated with the following code:
    ```rust
    dataflow i16 i, i16 j in [0:N, 0] {
        multistream<f32> bcast = broadcast_stream(0, 0) {
            channels = auto
        }
    }

    compute i16 i, i16 j in [0:N, 0] {
        await broadcast(a, bcast);
    }
    ```

In the future the functionality could be extended with an optional send-receive routing (like in the reduce case) for devices that do not support broadcast communication.

## Reduce

Most architectures do not support Reduce operations. Therefore we translate reduces to simple send-receive communication.

```
NOTE: We currently only support reduce in a N-by-N grid 
that can not be defined partially or in multiple rounds.
```

In the dataflowblock a reduce is defined the following way:
```
multistream<i16> name = reduce_stream(root_x, root_y) {
            graph = auto,
            op = S_SUM,
            pipelined = true
        }
```
where (root_x, root_y) defines the receiver. The name is important to give as an argument in the compute blocks. graph chooses the layout the communication follows. Further details on the different layouts available can be found in the [Layouts section](layouts.md). op defines which operation to use for the reduce. The currently supported list can be found below. pipelined can either be 'true' or 'false' and defines whether when sending arrays the whole array gets received by the next processing element (pipelined = false) or if each element of the array gets send on before receiving the next element.

The options for the operation `op` are:

- CL_SUM (returns the sum of all elements)
- CL_PRODUCT (returns the product of all elements)

At the moment parameters are not allowed in the range of the coordinate grid, i.e.
```rust
dataflow i16 i, i16 j in [0:N , 0:N] {...}
```
is not allowed.

In the computeblocks a reduce can then be used with the following line:
```rust
await reduce(data, name)
```
where data is the element/array to reduce on.

???+ example "Example: Simple Broadcast"
    ```rust
    result
    ```

    can be generated from:

    ```rust
    input
    ```