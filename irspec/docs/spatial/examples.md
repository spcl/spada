
## Vertical Advection


```rust
kernel vadv<I,J,K>(stream<f32>[I, J] utens_stage,
                  stream<f32>[I, J] readonly u_stage,
                  stream<f32>[I, J] readonly wcon,
                  stream<f32>[I, J] readonly u_pos,
                  stream<f32>[I, J] readonly utens,
                  stream<f32>[I, J] writeonly datacol,       
                  f32 compiletime dtr_stage,
                  f32 compiletime bet_m,
                  f32 compiletime bet_p) {
  
  ////
  // Read Inputs

  // A place block on the outer scope carries data between the input phase
  // and the computation phase.
  place i16 i, i16 j in [0:I, 0:J] {
      # Inputs & Outputs arrays
      f32[K] utens_stage_l;
      f32[K] utens_stage_l;
      f32[K] u_stage_l;
      f32[K] wcon_l;
      f32[K] u_pos_l;
      f32[K] utens_l;
      f32[K] datacol_l;
  }
  
  phase {
    
    // Copy the data
    compute i16 i, i16 j in [0:I, 0:J] {
        foreach k, x in [0:K], receive(u_stage[i, j]) {
            u_stage_l[k] = x;
        }
        foreach k, x in [0:K], receive(wcon[i, j]) {
            wcon_l[k] = x;
        }
        foreach k, x in [0:K], receive(u_pos[i, j]) {
            u_pos_l[k] = x;
        }
        foreach k, x in [0:K], receive(utens[i, j]) {
            utens_l[k] = x;
        }
        foreach k, x in [0:K], receive(utens_stage[i, j]) {
            utens_stage_l[k] = x;
        }
        // Exploits implicit completions at the end of each phase
    }
  }
  
  ////
  // Computation

  phase {
  
    place i16 i, i16 j in [0:I, 0:J] {
      # Local fields live inside the phases scope
      f32[K] gav;
      f32[K] gcv;
      f32[K] as_;
      f32[K] cs;
      f32[K] acol;
      f32[K] ccol;
      f32[K] bcol;
      f32[K] correction_term;
      f32[K] dcol;
      f32[K] ccol_2;
      f32[K] dcol_2;
      f32[K] datacol_l;
      f32 divided;
    }
  
    // Set up communication streams
    // The only PE-PE communication is for wcon, which is sent to the west  
    // We additionally set up a stream for the outputs
    dataflow i16 i, i16 j in [0:I, 0:J] {
      stream<f32> westwards = relative_stream(-1, 0);
    }
  
    ////
    // Computation
  
    compute i16 i, i16 j in [0, 0:J] {
      // Boundary condition
      // ...
      foreach k, x in [0:K], receive(westwards) {
        // ...
      }
    }
  
    compute i16 i, i16 j in [1:I, 0:J] {
        
        completion c1 = send(wcon_local[0:1], westwards);
        completion c2 = send(wcon_local[1:K], westwards);

        // base of the forward
        await foreach i32 k, f32 x in [0:1], receive(westwards) {
            gav[k] = -0.25 * x * wcon_l[k];
            gcv[k] = 0
            // ...
        }

        // Forward pass: data movement
        await foreach i32 k, f32 x in [1:K], receive(westwards) {
          gav[k] = -0.25 * x * wcon_l[k];
          gcv[k-1] = 0.25 * x * wcon_l[k];

          as_[k] = gav[k] * bet_m;
          cs[k] = gcv[k] * bet_m;
          acol[k] = gav[k] * bet_p;
          ccol[k] = gcv[k] * bet_p;
          bcol[k] = dtr_stage - acol[k] - ccol[k];

          correction_term[k] = -as_[k] * (u_stage_l[k-1] - u_stage_l[k]) - cs[k] * (u_stage_l[k+1] - u_stage_l[k]);
          dcol[k] = dtr_stage * u_pos_l[k] + utens_l[k] + utens_stage_l[k] + correction_term[k];

          // Thomas forward
          divided = 1.0 / (bcol[k] - ccol[k-1] * acol[k]);
          ccol_2[k] = ccol[k] * divided;
          dcol_2[k] = (dcol[k] - dcol[k-1] * acol[k]) * divided;
        }
  
        // Boundary condition (k=K-1) not shown
        for i32 k in [K-1] {
          /// Boundary condition ...
        }
        // Main backwards loop          
        for i32 k in [K-2:0:-1] {
          datacol_l[k] = dcol_2[k] - ccol_2[k] * datacol_l[k+1];
          utens_stage_l[k] = dtr_stage * (datacol_l[k] - u_pos_l[k]);
        }
  
        await c1
        await c2
  
        // Copy the data to the output
        await send(datacol_l, datacol[i, j]);
        await send(utens_stage_l, utens_stage[i, j]);
    }
  }

}


```


## 2D Laplacian

```rust
kernel laplacian<I,J,K> (stream<f32>[I+2, J+2] readonly in_field,
                         stream<f32>[I, J] writeonly lap_field) {
    
  ////////////////////////////////
  // Data placement
  

  
  place i16 i, i16 j in [0:I+2, 0:J+2] {
      f32[K] local_input;
  }
  
  place i16 i, i16 j in [1:I+1, 1:J+1] {
      f32[K] local_result;
  }
  
  /// Copy input data
  
  phase {
    compute i16 i, i16 j in [0:I+2, 0:J+2] {
      await receive(lap_field[i, j], local_input);
    }
  }

  phase {
    // Set up communication streams
    dataflow i16 i, i16 j in [0:I+2, 0:J+2] {
       stream<f32> eastwards = relative_stream(+1, 0);
       stream<f32> westwards = relative_stream(-1, 0);
       stream<f32> northwards = relative_stream(0, -1);
       stream<f32> southwards = relative_stream(0, +1);
       
    }

    // Edge senders
    compute i16 i, i16 j in [0, 1:J] {
        // Streaming send to the right
        completion c = send(local_input, eastwards);
        // We receive nothing
    }
    
    compute i16 i, i16 j in [I+1, 1:J] {
        // Streaming send to the left
        completion c = send(tosend, westwards);
        // We receive nothing
    }
    // ...
  
    compute i16 i, i16 j in [1:I+1, 1:J+1] {
        // Streaming parallel computation (map)
        completion f = map i32 k in [0:K] {
            local_result[k] = local_input[k] * 4;
        }
  
        // No data race, both map and send are reading from local_input
        completion w1 = send(local_input, westwards);

        // Example of an await for a completion.
        await f;
        
        // Writing to local_result form the map would be considered a data race
        // if we did not await f
        await foreach i32 k, f32 x in [0:K], receive(westwards) {
          local_result[k] = local_result[k] - x;
        }

        // Writing to the same array from multiple foreach blocks concurrently
        // is considered a data race.
        // Hence, we need to run one after the other.
        completion e1 = send(local_input, eastwards);
        await foreach i32 k, f32 x in [0:K], receive(eastwards) {
          local_result[k] = local_result[k] - x;
        }
        // ...

        // after all computation has finished:
        await send(local_result, lap_field[i, j]);   
    }  
  }
}
```

## Horizontal Diffusion

This example demonstrates how to turn horizontal diffusion into Spatial IR.
```python title="Horizontal Diffusion in GT4Py"
def horizontal_diffusion(in_field: Field3D, out_field: Field3D,
                          coeff: Field3D):
     with computation(PARALLEL), interval(...):
         lap_field = 4.0 * in_field[0, 0, 0] - (in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0])
         res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
         flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res
         res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
         fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res
         out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
             flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] -
             fly_field[0, -1, 0])
```


```rust title="Horizontal Diffusion in Spatial IR"
kernel <I, J, K>hdiff(stream<f32>[I+2, J+2] readonly in_stream,
                      stream<f32>[I, J] writeonly out_stream,
                      stream<f32>[I, J] readonly coeff_stream) {

    // Data placement
    place i16 i, i16 j in [0:I+1, 0:J+1] {
        f32[K] in_field;
        // in field of the east neighbor
        f32[K] in_field_east;
        // in field of the south neighbor 
        f32[K] in_field_south;
        f32[K] out_field;
        f32[K] coeff;
        f32[K] lap_field;
        f32[K] res;
        f32[K] flx_field;
        f32[K] fly_field;
    }
    
    // Read input
    phase {
        compute i16 i, i16 j in [1:I, 1:J] {
           await receive(coeff, coeff_stream[i-1, j-1]);
        }
        compute i16 i, i16 j in [0:I+1, 0:J+1] {
            await receive(in_field, in_stream[i, j]);
        }
    }
    
    // Laplacian computation
    // There is a small difference to the indipendent 2D laplace example
    // We need to store the in_field of the east and south neighbor
    phase {
    
        // Set up communication streams
        dataflow i16 i, i16 j in [0:I+1, 0:J+1] {
            stream<f32> eastwards = relative_stream(1, 0);
            stream<f32> westwards = relative_stream(-1, 0);
            stream<f32> northwards = relative_stream(0, -1);
            stream<f32> southwards = relative_stream(0, 1);
        }
    
        // See the 2D laplacian example
        // We store in_field_east and in_field_south

        compute i16 i, i16 j in [1:I, 1:J] {
    
            completion f = map i32 k in [0:K] {
                lap_field[k] = in_field[k] * 4;
            }
      
            completion c = send(lap_field, westwards);
    
            await f;
            
            // Result needed: store in in_field_east
            await foreach i32 k, f32 x in [0:K], receive(westwards) {
                in_field_east[k] = x;
            }
            // Then, decrement the lap_field using the stored in_field_east
            await map i32 k in [0:K] {
                lap_field[k] = lap_field[k] - in_field_east[k];
            }
            
            await c;
    
            completion c2 = send(local_input, eastwards);
            
            // result not needed, decrement directly.
            await foreach i32 k, f32 x in [0:K], receive(eastwards) {
                lap_field[k] = lap_field[k] - x;
            }
            
            await c2;
            // ...
        }
       
        // Boundary conditions (omitted)
    }
    
    phase {
        // Set up streams to communicate laplacian results
        // Note that we could re-use the same streams as in the previous phase
        // And we could also merge the two phases into one
    
        dataflow i16 i, i16 j in [0:I+1, 0:J+1] {
            // Note that the directions are reversed compared to gt4py because
            // we specify the offsets to send to, wheres it specifies the offsets to read from
            stream<f32> west = relative_stream(-1, 0);
            stream<f32> north = relative_stream(0, -1);
            stream<f32> flx_field_stream = relative_stream(1, 0);
            stream<f32> fly_field_stream = relative_stream(0, 1);
        }
        
        compute i16 i, i16 j in [1:I-1, 1:J-1] {
            
            completion c = send(lap_field, west);
            
            foreach i32 k, f32 x in [0:K], receive(west) {
                res[k] = x - lap_field[k];
            }
            
            await c;
            
            await map i32 k in [0:K] {
                flx_field[k] = ((res[k] * (in_field_east[k] - in_field[k])) <= 0) * res[k];
            }
            
            completion c3 = send(lap_field, north);
            
            foreach i32 k, f32 y in [0:K], receive(north) {
                res[k] = y - lap_field[k];
            }
            
            await c3;
            
            await map i32 k in [0:K] {
                fly_field[k] = ((res[k] * (in_field_south[k] - in_field[k])) <= 0) * res[k];
            }
            
            completion c4 = send(flx_field, flx_field_stream);

            // No need to copy the flx_field, can accumulate directly
            await foreach i32 k, f32 x in [0:K], receive(flx_field_stream) {
                out_field[k] = flx_field[k] - x;
            }
            
            await c4;
            
            completion c5 = send(fly_field, fly_field_stream);
            
            // No need to copy the fly_field, can accumulate directly
            await foreach i32 k, f32 y in [0:K], receive(fly_field_stream) {
                out_field[k] = outfield[k] + fly_field[k] - y;
            }
            
            await map i32 k in [0:K] {
                out_field[k] = in_field[k] - local_coeff[k] * out_field[k];
            }
            
            // We may overlap sending of fly with the computation of out_field
            await c5;
            
            await send(out_field, out_stream[i-1, j-1);
        }
        
        compute i16 i, i16 j in [0, 1:J-1] {
            // Boundary condition
            // ...
        }
        
        compute i16 i, i16 j in [I, 1:J-1] {
            // Boundary condition
            // ...
        }
        
        compute i16 i, i16 j in [1:I-1, 0] {
            // Boundary condition
            // ...
        }
        
        compute i16 i, i16 j in [1:I-1, J] {
            // Boundary condition
            // ...
        }
    }
}

```


## Streaming 1D Convolution

Performs the convolution of a 1D kernel with a streaming K-D input array that changes over time.
That is in each time step, we receive an array of K elements, and we convolve it with the kernel.

The kernel is of size 3.
While the data is being streamed, it is convolved with the kernel and streamed to the output.

```rust
kernel conv<J>(stream<f32>[J] readonly input,
               stream<f32>[J] writeonly output,
               f32[3] readonly kernel) {

    // Data placement
    place i16 i, i16 j in [0:J, 0] {
        f32 y;
    }

    // Communication
    dataflow i, j in [0:J, 0] {
        stream<f32> eastwards = relative_stream(1, 0);
        stream<f32> westwards = relative_stream(-1, 0);
    }

    // Computation
    compute i16 i, i16 j in [1:J-1, 0] {

        // Streaming receive
        // Each PE receives a single scalar per time step
        foreach f32 x in receive(input[i]) {

            y = x * kernel[1];

            // Send the data to the right
            completion comp_west = send(x, westwards);

            await foreach i16 k, f32 x2 in [0:1], receive(westwards) {
              y = y + x2 * kernel[0];
            }

            await comp_west;

            // Send the data to the left
            completion comp_east = send(x, eastwards);

            await foreach i16 k, f32 x3 in [0:1], receive(eastwards) {
               y = y + x3 * kernel[2];
            }

            await comp_east;

            // Send the result to the output
            await send(y, output[i]);
        }
    }

    // Left corner
    compute i16 i, i16 j in [0, 0] {
        // Streaming receive
        foreach f32 x in receive(input[i]) {
            // Send the data to the left
            // S1
            completion comp_east = send(x, eastwards);
            // S2
            y = x * kernel[1];
            // S3
            await foreach i16 x, f32 y in [0:1], receive(eastwards) {
                // S4
                y = y + x * kernel[2];
            }
            // S5
            await comp_east;
            
            // S6
            // Send the result to the output
            await send(y, output_local);
        }
    }

    // Right corner
    // ...

}
```
