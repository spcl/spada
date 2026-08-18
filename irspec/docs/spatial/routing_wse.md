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
    real loop and so costs code and compile time independent of the number of rounds. Compare
    `samples/spatial/sorting/odd_even_sort_1D.sptl` with
    `samples/spatial/sorting/odd_even_sort_1D_looped.sptl`.

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
    two hops through a plain relay, and two switch-configured routers in sequence. The compiler
    therefore emits `encode_single_payload`, which writes slot 0 only.

    The consequence is that a message cannot advance one router while leaving another on the same
    path where it is. *If the routers along one path would have to advance by different amounts, a
    compile error is raised.* A router that is already on its last configuration is exempt: it never
    routes anything again, so a message passing through may over-advance it harmlessly.

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
    message and nothing overshoots. `samples/spatial/sorting/odd_even_sort_1D.sptl` avoids it by
    giving each round parity its own pair of channels: each PE's role on a channel is then fixed,
    no router switches at all, and no close emits a message.

Because the control message travels the path of the retired configuration in order behind the data,
a receiving PE needs to emit nothing to advance its own router: the ordering required by the
[lemma](../routing#undefined-behavior) in the IR semantics is provided by the fabric. A receiver's `close` 
therefore has no runtime effect; it exists so that the lifetime of the stream — and hence the number
of elements it carries — is stated by every participant and can be checked.
