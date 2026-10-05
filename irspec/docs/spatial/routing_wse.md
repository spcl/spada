# Lowering Routing Declarations to Cerebras Wafer-Scale Engine

This page describes how the **Cerebras WSE / CSL backend** realizes routing concepts, such as epochs,
on the target hardware architecture. The SpaDA IR abstracts away details such as colors, routers,
switch positions, and control wavelets, and the code generation process lowers to the specific
WSE architecture selected via the `WSE_ARCH` environment variable (`wse2` or `wse3`). The code
generator preserves the correctness conditions defined in
[Undefined Behavior](../routing#undefined-behavior).

## Lowering to Switches

Fabric channels correspond to hardware routing colors. Epochs enable the sequential reuse of channels
and colors across multiple communication phases.

A stream induces a *route configuration* at each PE along its path, defined by an input port set and an
output port set (where `RAMP` denotes the local compute element). For a fixed channel $C$ and PE
$(i, j)$, ordering the active streams across successive epochs yields a sequence of configurations
$R_0, R_1, \dotsc, R_{n-1}$. This sequence maps directly to the router's hardware switch positions
for the assigned color: $R_0$ is the base configuration, and the router advances to $R_{k+1}$ at
epoch boundaries.

Unspecified directions in a switch position retain their previous configuration.

!!! warning "WSE-2: Single-Direction Switch Positions"
    On WSE-2, each switch position may configure either an input direction or an output direction,
    not both simultaneously. A transition changing both directions (such as a PE transitioning from
    receiver to sender) requires two switch positions and an intermediate relay configuration.

    WSE-3 supports configuring both input and output directions within a single switch position,
    requiring only one position and one advance for bidirectional transitions.

!!! danger "Error: Switch Position Capacity Exceeded"
    Routers provide up to four switch positions per color on both WSE-2 and WSE-3. If the stream
    sequence on a channel requires more than four positions at any PE, compilation fails. Such
    cases must be resolved by assigning additional channels to partition the traffic.

When the sequence of configurations at a router is periodic, only one period is stored and the router
is configured with `ring_mode = true` to wrap back to the initial position.

Consecutive configurations that are identical do not consume switch positions and require no advance.

!!! tip "Phase Elimination for Static Configurations"
    When a channel's route configuration remains constant across all epochs, its routers never
    advance. In such cases, epoch boundaries enforce only relative ordering. A sequence of
    identical-configuration phases can be collapsed into a single epoch containing sequential loops
    (`for`) within compute blocks, lowering directly to CSL loops and eliminating phase synchronization
    barriers. Examples include:
    - `samples/spatial/sort/odd_even_sort_1D_looped.sptl`: $N$ odd-even rounds over four static
      channels within a single CSL loop.
    - `samples/spatial/sort/shearsort_2D.sptl`: 2D mesh sort using eight static channels across
      nested loops without dynamic switches. Interior PEs receive on four colors within a single
      epoch, requiring WSE-3 (six input queues).

    This optimization applies only to non-switching channels, as dynamic switch sequences require
    compile-time epoch boundaries.

Router advances are triggered by stream `close` operations at epoch boundaries:
- **Remote advance**: When downstream routers along the path must advance, the sender transmits a
  switch-advance control wavelet along the channel using the outgoing configuration. The control
  wavelet advances each traversed router after all data wavelets have cleared.
- **Local advance (WSE-2)**: When only the sender's own router must advance by a single position,
  the final data wavelet triggers the transition via `.advance_switch` on the fabric output DSD,
  avoiding control wavelet transmission and preventing output queue contention.
- **Local advance (WSE-3)**: WSE-3 emits a `SWITCH_ADV` control wavelet, as queue depth and
  queue-to-color binding semantics accommodate explicit control wavelets.

!!! warning "Uniform Switch Advances Along a Path"
    CSL control wavelets contain command slots for router reconfiguration (`<control>'s MAX_CMDS`).
    In hardware execution and simulation, only command slot 0 is executed by traversed routers.
    Every switch-configured router reached by the control wavelet applies the command in slot 0.

    Consequently, a single control wavelet cannot selectively advance a subset of routers along a
    path while leaving others unchanged. All advancing routers along an active path must advance by
    an identical number of positions; otherwise, a compile error is raised. Routers that have reached
    their final switch position are exempt: outside `ring_mode`, advances past the final position are
    no-ops.

!!! danger "WSE-2: Turnaround Control Wavelet Propagation"
    A switch-advance control wavelet terminates at the first router whose active routing configuration
    targets `RAMP`. In WSE-2 bidirectional turnarounds (where a receiver transitions to become a
    sender), the two required switch positions can cause the second control message to route past
    the intermediate receiver to downstream PEs.

    To avoid this issue on WSE-2, communication topologies where PEs alternate sending and receiving
    roles on a shared color should be partitioned into separate directional channels (e.g.,
    `samples/spatial/sort/odd_even_sort_1D_looped.sptl`).

Because switch-advance control wavelets traverse the path behind the payload data, receiving PEs do
not need to emit messages to advance their routers. A receiver's `close` statement has no runtime
overhead; it serves to validate stream lifetimes and element transfer counts statically.

## Overlapping Interval Shifts

When multiple consecutive PEs shift data by uniform distance $d$ along an axis:

```
dataflow i16 i, i16 j in [0:D + M, 0] {
  stream<f32, 1> fwd = relative_stream(D, 0) { hops = auto, channel = 0 }
}
```

with sources in $[0:M)$ and destinations in $[D:D+M)$, communication paths overlap along intermediate
routers. Without optimization, this pattern requires either $M$ distinct channels or a serialized
store-and-forward chain.

The compiler detects this pattern (`detect_shift_bundles`) and lowers the entire shift onto **a single
channel** using coordinated router switching and destination filtering:

```
PE:        0      1      2      3         4         5      (M = 3, D = 3)
role:      src0   src1   src2   dst0      dst1      dst2
sends:     3rd    2nd    1st    --        --        --
routes:    R->E   R->E   R->E   W->{R,E}  W->{R,E}  W->R
pos1:      --     W->E   W->E   --        --        --
filter:    --     --     --     win 2     win 1     win 0
```

### Mechanism

1. **Descending Transmission Order**: Sources transmit in descending order of distance to the
   destination range (nearest source first). Hardware link backpressure automatically serializes
   transfers without software coordination.
2. **Local Source Advance**: Each source initializes with transmission from the local ramp
   (`rx = RAMP, tx = {EAST}`). Upon completing its own transmission, the source locally advances
   its router to relay mode (`rx = WEST, tx = {EAST}`). On WSE-2, this advance is executed via
   `.advance_switch` on the final data DSD; on WSE-3, via `SWITCH_ADV`.
3. **Static Destination Filtering**: Destination routers do not switch during the epoch. Intermediate
   destinations duplicate traffic to both the local ramp and downstream neighbors (`tx = {RAMP, EAST}`),
   while the terminal destination consumes the stream (`tx = {RAMP}`). Each destination isolates its
   designated slice of data using a hardware counter filter.

!!! note "Counter Filter Configuration"
    A hardware counter filter initializes at `init_counter`, increments on every counted data wavelet,
    and resets to zero after reaching `limit1`. A wavelet is delivered to the local compute element
    if and only if `counter <= max_counter`. For a window of $W$ words within a total stream of
    $L \times W$ words:
    $$\text{limit1} = L \cdot W - 1, \quad \text{max\_counter} = W - 1$$
    `init_counter` is configured as an affine function of PE coordinates such that the counter
    reaches zero at the arrival of the target block.

!!! danger "Hardware Filter Capacity"
    WSE-2 and WSE-3 provide four hardware filters per PE, of which one is reserved by the `memcpy`
    runtime module, leaving **three** available for application kernels (`FILTERS_PER_PE`). Because
    hardware filters cannot be safely reconfigured while traffic is active, filters are configured
    once at layout time and cannot be reused across phases. Kernels requiring more than three filtered
    phases must allocate separate channels.

### Bundling Preconditions and Channel Reuse

Shift bundling applies when every decomposed contiguous segment contains at least two sources and
does not exceed the shift distance ($2 \le M \le D$). Shifts of distance 1 are mapped directly to
standard switch chains.

Unbundled shifts may safely share a channel across phases when they share axis, distance $d$, and
source coordinates modulo $2d$. Because source, destination, and relay roles form disjoint residue
classes modulo $2d$, router configurations remain invariant across all pooled phases.

Channel sharing can be extended across opposite directions along the same axis by pooling by distance
and residue modulo $2d$. Each PE maintains a single role on the color across all phases, requiring at
most two switch positions (one for transmitting direction, one for receiving direction).
