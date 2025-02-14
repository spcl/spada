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

???+ example "Example: Simple Broadcast with snake communication"
    ```rust
    kernel @add<N, K>() {
        place i16 i, i16 j in [0:5 , 0:5] {
            i16[100] a
        }
        dataflow i16 i, i16 j in [0:5:1 , 0:5:1] {
            stream<i16> reduce = relative_stream(-1, 0)
        }
        dataflow i16 i, i16 j in [0:5:1 , 1:4:1] {
            stream<i16> reduce#1 = relative_stream(1, 0)
        }
        dataflow i16 i, i16 j in [0:1:1 , 1:5:1] {
            stream<i16> reduce#2 = relative_stream(0, -1)
        }
        dataflow i16 i, i16 j in [4:5:1 , 0:4:1] {
            stream<i16> reduce#3 = relative_stream(0, -1)
        }
        compute i16 i, i16 j in [0:1 , 0:1] {
            a[0] = 1
            await foreach i32 reduce_runner, i16 reduce_receive in [0:100], receive(reduce#1) {
            a[reduce_runner] = (a[reduce_runner] + reduce_receive)
            }
        }
        compute i16 i, i16 j in [0:1 , 1:2] {
            a[0] = 1
            await foreach i32 reduce_runner#1, i16 reduce_receive#1 in [0:100], receive(reduce#1) {
            a[reduce_runner#1] = (a[reduce_runner#1] + reduce_receive#1)
            }
            await send(a, reduce#2)
        }
        compute i16 i, i16 j in [0:1 , 2:3] {
            a[0] = 1
            await foreach i32 reduce_runner#2, i16 reduce_receive#2 in [0:100], receive(reduce#1) {
            a[reduce_runner#2] = (a[reduce_runner#2] + reduce_receive#2)
            }
            await send(a, reduce#2)
        }
        compute i16 i, i16 j in [0:1 , 3:4] {
            a[0] = 1
            await foreach i32 reduce_runner#3, i16 reduce_receive#3 in [0:100], receive(reduce#1) {
            a[reduce_runner#3] = (a[reduce_runner#3] + reduce_receive#3)
            }
            await send(a, reduce#2)
        }
        compute i16 i, i16 j in [0:1 , 4:5] {
            a[0] = 1
            await foreach i32 reduce_runner#4, i16 reduce_receive#4 in [0:100], receive(reduce#1) {
            a[reduce_runner#4] = (a[reduce_runner#4] + reduce_receive#4)
            }
            await send(a, reduce#3)
        }
        compute i16 i, i16 j in [4:5 , 0:1] {
            a[0] = 1
            await foreach i32 reduce_runner#5, i16 reduce_receive#5 in [0:100], receive(reduce#4) {
            a[reduce_runner#5] = (a[reduce_runner#5] + reduce_receive#5)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [4:5 , 1:2] {
            a[0] = 1
            await foreach i32 reduce_runner#6, i16 reduce_receive#6 in [0:100], receive(reduce#2) {
            a[reduce_runner#6] = (a[reduce_runner#6] + reduce_receive#6)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [4:5 , 2:3] {
            a[0] = 1
            await foreach i32 reduce_runner#7, i16 reduce_receive#7 in [0:100], receive(reduce#2) {
            a[reduce_runner#7] = (a[reduce_runner#7] + reduce_receive#7)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [4:5 , 3:4] {
            a[0] = 1
            await foreach i32 reduce_runner#8, i16 reduce_receive#8 in [0:100], receive(reduce#2) {
            a[reduce_runner#8] = (a[reduce_runner#8] + reduce_receive#8)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [4:5 , 4:5] {
            a[0] = 1
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [1:4 , 0:1] {
            a[0] = 1
            await foreach i32 reduce_runner#9, i16 reduce_receive#9 in [0:100], receive(reduce#1) {
            a[reduce_runner#9] = (a[reduce_runner#9] + reduce_receive#9)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [1:4 , 1:2] {
            a[0] = 1
            await foreach i32 reduce_runner#10, i16 reduce_receive#10 in [0:100], receive(reduce#1) {
            a[reduce_runner#10] = (a[reduce_runner#10] + reduce_receive#10)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [1:4 , 2:3] {
            a[0] = 1
            await foreach i32 reduce_runner#11, i16 reduce_receive#11 in [0:100], receive(reduce#1) {
            a[reduce_runner#11] = (a[reduce_runner#11] + reduce_receive#11)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [1:4 , 3:4] {
            a[0] = 1
            await foreach i32 reduce_runner#12, i16 reduce_receive#12 in [0:100], receive(reduce#1) {
            a[reduce_runner#12] = (a[reduce_runner#12] + reduce_receive#12)
            }
            await send(a, reduce#1)
        }
        compute i16 i, i16 j in [1:4 , 4:5] {
            a[0] = 1
            await foreach i32 reduce_runner#13, i16 reduce_receive#13 in [0:100], receive(reduce#1) {
            a[reduce_runner#13] = (a[reduce_runner#13] + reduce_receive#13)
            }
            await send(a, reduce#1)
        }
    }
    ```

    can be generated from:

    ```rust
    kernel @add<N,K>() {

        place i16 i, i16 j in [0:5, 0:5] {
            i16[100] a
        }

        dataflow i16 i, i16 j in [0:5, 0:5] {
            multistream<i16> red = reduce_stream(0, 0) {
                algorithm = snake,
                op = CL_SUM,
                pipelined = false
            }
        }

        compute i16 i, i16 j in [0:5, 0:5] {
            a[0] = 1
            await reduce(a, red)
        }
    }
    ```

    The layout for this example can be found as the snake example in the [Layouts section](layouts.md).