# Lowering Routing Declarations to Cerebras Wafer-Scale Engine

This page describes how the **Cerebras WSE / CSL backend** realizes routing concepts, such as epochs,
on the target hardware architecture. The SpaDA IR abstracts away details such as colors, routers,
switch positions, and control wavelets, and the code generation process lowers potentially differently
to specific WSE architecture (selected via the `WSE_ARCH` environment variable). Where
the two WSE generations differ, the text says so; the compiler selects between them on
`WSE_ARCH`. The code generator still preserves the correctness conditions of
[Undefined Behavior](../routing#undefined-behavior). 

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

Whichever side a switch position leaves unspecified keeps the value it currently has, so positions
compose incrementally.

!!! warning "WSE-2: A Switch Position Carries One Direction"
    On WSE-2 a switch position records *either* the input the router receives from or the output it
    transmits to, never both — `cslc` rejects a position naming both with *"cannot have both an
    input and an output in the same switch position"*. A transition that changes both sides — a PE
    that stops receiving on a channel and starts sending on it, as in a systolic chain — therefore
    occupies **two** positions, passing through an intermediate configuration that keeps the old
    input and takes the new output. The intermediate is a pure relay, occupied only between the two
    advances that retire the configuration, and it must keep the old input so that the second
    advance still reaches the router.

    WSE-3 accepts both directions in one position, so the same transition costs one position and one
    advance there.

!!! danger "Error: Too Many Route Configurations"
    A router holds a bounded number of switch positions per color (four on both WSE-2 and WSE-3).
    *If the streams sharing a channel require more positions than that at a single PE, a compile
    error is raised.* Assigning a different channel to some of the streams resolves it, at the cost
    of an additional color. On WSE-2 a configuration which changes both the input and the output
    direction costs two positions, so four positions is fewer than four turnarounds there.

When the sequence of configurations at a router is periodic — a halo exchange that alternates
between sending and receiving across phases produces $R_0, R_1, R_0, R_1$ — only one period is
stored and the switch wraps around from the last position back to the base one (`ring_mode`).

Consecutive configurations that are equal do not consume a position and do not require an advance.
This is a common case: two streams declared as `relative_stream(-2, 0)` in successive phases induce
the same configuration at every PE of their paths, so their shared channel needs no switching at
all.

!!! tip "When the Configuration Never Changes, the Phases May Not Be Needed"
    A channel whose route configuration is the same in every epoch is never reassigned, and its
    routers never advance. The epoch boundaries around it are then buying only *ordering* — and the
    fabric already delivers a channel's wavelets in order. Such a sequence of phases can be
    collapsed into a single epoch with a sequential `for` in the compute blocks, which lowers to a
    real loop and so costs code and compile time independent of the number of rounds.
    `samples/spatial/sort/odd_even_sort_1D_looped.sptl` is the example: N odd-even rounds on four
    static channels, one CSL loop, no per-round barrier.

    This does *not* generalize to channels that switch: a router's positions are a static sequence,
    so the epoch a configuration belongs to has to be visible to the compiler.

An advance is driven by the `close` that ends the epoch. The sending PE emits a *switch-advance
control message* on the channel, one per position to be traversed. It follows the stream's path
using the configuration that is being retired, and advances the router of each PE it traverses,
after all data of the epoch.

!!! warning "WSE: Advances Are Not Selective"
    A CSL control wavelet nominally carries up to eight per-router switching commands
    (`<control>`'s `MAX_CMDS`), which would let one message advance some routers on a path and leave
    others alone. **On the WSE hardware it does not work that way.** Measured on the simulator, only
    command slot 0 is ever executed, and **every** switch-configured router the wavelet reaches
    applies it; slots 1–7 had no effect in any topology tested — the sender's own router, one hop,
    two hops through a plain relay, and two switch-configured routers in sequence. A generated kernel
    built on the opposite assumption, advancing the fourth router of a path with an `[ADV, NOP, NOP,
    ADV]` chain, stalled in the fabric. The compiler therefore emits `encode_single_payload`, which
    writes slot 0 only.

    The consequence is that a message cannot advance one router while leaving another on the same
    path where it is. *If the routers along one path would have to advance by different amounts, a
    compile error is raised.* A router that is already on its last position is exempt: outside
    `ring_mode` an advance past the last position is a no-op, so a message passing through may
    over-advance it harmlessly. (Such a router is not necessarily finished — it may keep relaying the
    same configuration for the rest of the kernel, which is exactly what the bundle below relies
    on.)

    This is a property of the *payload*, not of switching: a router can still be switched at a time
    only it knows, and delivery to a compute element can still be made selective, by the two
    mechanisms the next section combines.

!!! danger "WSE-2: A Two-Advance Turnaround Overshoots the Receiver"
    A control message stops at the first router whose current output is `RAMP`, and it is routed by
    the position that router holds *when the message arrives*. The two messages of a WSE-2
    turnaround therefore do not travel the same distance: the first stops at the receiver and moves
    it onto the intermediate position, whose output is a real direction rather than `RAMP`, so the
    second is *forwarded past the receiver* to the next PE on the line.

    That is harmless when the next PE is not switch-configured on the same color. When it is, the
    stray message reaches a router that no epoch of its own is retiring. **The compiler does not
    detect this**, and a kernel that trips it hangs rather than failing to compile.

    An odd-even transposition sort makes the hazard concrete: put both round parities on one
    eastward channel and every interior PE alternates between sending and receiving on it, so every
    close is a turnaround and every receiver has a switch-configured neighbour behind it. That
    kernel deadlocks on WSE-2 and runs on WSE-3, where the turnaround costs one position and one
    message and nothing overshoots. `samples/spatial/sort/odd_even_sort_1D_looped.sptl` avoids it
    by giving each round parity its own pair of channels: each PE's role on a channel is then
    fixed, no router switches at all, and no close emits a message.

Because the control message travels the path of the retired configuration in order behind the data,
a receiving PE needs to emit nothing to advance its own router: the ordering required by the
[lemma](../routing#undefined-behavior) in the IR semantics is provided by the fabric. A receiver's `close` 
therefore has no runtime effect; it exists so that the lifetime of the stream — and hence the number
of elements it carries — is stated by every participant and can be checked.

## Overlapping Interval Shifts

The [correctness conditions](../routing#undefined-behavior) rule out one shape that occurs constantly:
a run of consecutive PEs all shifting the same distance $d$ along an axis, as in

```
dataflow i16 i, i16 j in [0:D + M, 0] {
  stream<f32, 1> fwd = relative_stream(D, 0) { hops = auto, channel = 0 }
}
```

with the PEs in `[0:M)` sending and those in `[D:D+M)` receiving. Source $p$'s word passes through
the routers of sources $p+1, \dotsc, M-1$, so the paths share PEs within one epoch. Written as one
stream per source, that is a channel each, $M$ colors for a shift; the alternative is a chain of
single-hop stores and forwards, which serializes the whole run behind $d$ hops of copying.

Neither is necessary. The compiler recognizes this pattern — `detect_shift_bundles` — and lowers the
whole run onto **one channel**, giving the routers configurations that are switched only by events
the PE owning them knows locally:

```
PE:        0      1      2      3         4         5      (M = 3, D = 3)
role:      src0   src1   src2   dst0      dst1      dst2
sends:     3rd    2nd    1st    --        --        --
routes:    R->E   R->E   R->E   W->{R,E}  W->{R,E}  W->R
pos1:      --     W->E   W->E   --        --        --
filter:    --     --     --     win 2     win 1     win 0
```

**Sources inject, then relay.** A source starts at `rx = RAMP, tx = {EAST}` with `pos1` taking
`rx = WEST`, sends its own words, and its `close` advances its own router into relay mode. The
trigger is local — *"my own send is done"* — which is what makes it expressible at all, given that
the payload of the resulting message [selects nothing](#lowering-to-switches).

**The order is descending, and enforces itself.** The source nearest the destinations goes first. No
schedule or barrier is needed: a source further away cannot push a word through its neighbour's
router while that neighbour is still injecting from its ramp, so it waits on the link. Backpressure
serializes the run in exactly the order the switches expect.

**Destinations do not switch; a filter picks their words.** Each destination is statically routed to
`tx = {RAMP, EAST}`, which *duplicates* rather than consumes: every destination's router sees the
entire stream, in one order, and the one the stream reaches last uses `tx = {RAMP}` to take it out of
the network. Which words a destination hands to its compute element is decided by a counter filter
on that color, one linear function of the PE coordinate, so a single `@set_color_config` covers the
whole run. A control message passes such a router without being counted (`count_data = true`) and
without being filtered.

!!! note "Note: Counter Filter Arithmetic"
    As measured on the simulator, a counter filter starts at `init_counter`, increments on every data
    wavelet, wraps to zero after `limit1`, and hands a wavelet to the compute element iff the counter
    is at most `max_counter`. A window of `words` out of a stream of `length * words` is therefore
    `limit1 = length * words - 1`, `max_counter = words - 1`, and an `init_counter` chosen so that
    the counter reads zero as the wanted block arrives.
    `tests/csl_runtime/test_shift_bundle_filters.sh` is the hand-written layout this was measured
    with.

!!! danger "Error: Too Many Wavelet Filters"
    WSE-2 has four filters per PE, of which the `memcpy` module reserves one, so **three** are usable
    (`FILTERS_PER_PE`). A PE needs one per color it filters. *If a PE would need more, a compile error
    is raised*; the fix is to give some of the streams their own channels, which trades filters for
    colors. Three filters therefore means at most three bundled phases per PE, whatever the kernel:
    `batcher_oddeven_bundled_1D.sptl` would want ten at $2^4$ PEs and bundles only its three widest
    phases, which is where most of the colors are saved anyway.

    Reconfiguring a filter while wavelets are still in flight on its color is a data race, and the
    destination that terminates the stream cannot be reconfigured until the stream has drained,
    because its router is what removes the wavelets from the network. Filters are consequently set up
    once, at layout time, and never reused between phases.

Bundling applies only when every run it decomposes into has at least two sources and is no longer
than the shift distance, so that no PE is both a source and a destination; a shift of one PE is left
alone, since a chain at distance one is already sequenced by ordinary switch positions. Anything
else falls back to the per-hop lowering, and to the errors above if that conflicts.

Which shifts are bundled is decided by the channel assignment rather than by an attribute: a bundle
is what several overlapping matchings on *one* channel become, so giving each matching a channel of
its own is how a kernel declines the trade. What it then costs is colors, and those can be won back
by reusing a channel across phases. Two unbundled shifts may share one safely when they agree on
axis and signed distance and their sources agree modulo twice that distance, because a PE's role —
source, relay or destination — is then a function of its position modulo twice the distance alone,
so one static configuration serves every phase in the pool. `batcher_oddeven_bundled_1D.sptl` pools
on exactly this rule, and `batcher_oddeven_1D.sptl` is the same rule written out as arithmetic.

The agreement on the *sign* of the distance can be dropped without giving that up. Keep the axis, the
magnitude and the source residue modulo twice it, and let the direction of travel vary: the sources
are then the PEs congruent to the residue, the destinations those congruent to residue plus distance,
and the relays the classes strictly between on the one side or the other — three disjoint classes, so
a PE still holds one role on the color for the whole kernel and still never both sends and receives
on it. What varies is the side it faces, which is one switch position either way, since a source only
ever changes where it transmits and a destination only where it receives. This halves the colors a
pooled distance needs, and with them the queues, which is what
`batcher_oddeven_wse3_1D.sptl` is for: on WSE-3 a queue stays bound to its color for the whole
kernel, so what a PE can afford is not how many colors are live at once but how many it ever touches.

Sharing on a basis looser than either does risk a PE that sends on the color in one phase and receives
on it in another, which needs a two-sided switch change that a sender cannot drive on WSE-2 (see
[Lowering to Switches](#lowering-to-switches)), and nothing in the compiler currently rejects it.

This arrangement is the one used in Schnyder's *Distributed Sorting on the Cerebras Wafer-Scale
Engine* (fig. 7.6) for the 2D reduce-scatter, and is known to run on WSE-2.

!!! note "Note: Multiple Rounds on One Color"
    Two mechanisms are deliberately left unused, and are what to reach for if the four switch
    positions or three filters run out. `SWITCH_RST` restores the initial configuration of every
    router a message passes, which retires a whole path with one wavelet. Teardown-based
    reconfiguration reprograms the routers between rounds outright, which is the only known way to
    put an arbitrary *sequence* of sends and receives on one color: for a long enough sequence there
    is a PE for which no fixed cycle of switch positions exists. Until then, splitting the rounds
    across channels — as `bitonic_sort_1D.sptl` does — remains the per-kernel fallback.
