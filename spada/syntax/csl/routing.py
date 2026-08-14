"""
Routing and switch planning for CSL code generation.

A stream's routing declaration says which PEs its data traverses; this module turns that into the
``@set_color_config`` calls of ``layout.csl``. Each router holds, per color, one base route
configuration plus up to three *switch positions*, and advances between them when a switch-advance
control message passes through. Streams that share a channel, and streams that a PE both receives
and forwards, are what make a router need more than one configuration.

The pieces, in the order they appear below:

* :class:`RouteConfig` -- one ``.rx``/``.tx`` pair, the configuration of one router for one color.
* :class:`ColorSwitchPlan` -- the ordered configurations a router cycles through.
* :func:`set_color_config` / :func:`switch_advance_payload` -- the only places that produce
  ``@set_color_config`` and ``<control>`` text.
* :func:`declare_switch_advances` -- the fabric descriptors those control messages are sent through.
* :func:`collect_routes` -- the parametric routing graph, as per-rectangle layout code.
* :func:`plan_switch_advances` -- which routers each ``close`` has to advance.

See ``irspec/docs/spatial/routing.md`` for the semantics this implements.
"""
from dataclasses import dataclass, field
from io import StringIO

from spada.syntax.csl import constants
from spada.syntax.csl import statements as cslstmt
from spada.syntax.csl import structures as cslstruct
from spada.syntax.spatial_ir import analysis, stream_lifetime
from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir.canonicalization import PEBlock
from spada.syntax.spatial_ir.grid_geometry import Rectangle


@dataclass(frozen=True)
class RouteConfig:
    """
    The configuration of one router for one color: the directions it receives from and transmits to.

    ``RAMP`` denotes the PE's own compute element.
    """
    rx: tuple[str, ...]
    tx: tuple[str, ...]

    def as_csl(self) -> str:
        return '.{ .rx = .{%s}, .tx = .{%s} }' % (', '.join(self.rx), ', '.join(self.tx))

    def as_switch_position(self, previous: 'RouteConfig') -> str:
        """
        Renders this configuration as a ``.posN`` struct, relative to the configuration it replaces.

        Only the side that changes is written: whatever a position leaves out keeps the value it
        currently has, so positions compose incrementally. The ``rx`` of a switch position also
        accepts a single direction only, unlike the base configuration.
        """
        parts = []
        if self.rx != previous.rx:
            if len(self.rx) != 1:
                raise ValueError(f'A switch position can only receive from a single direction, got {self.rx}')
            parts.append('.rx = %s' % self.rx[0])
        if self.tx != previous.tx or not parts:
            parts.append('.tx = .{%s}' % ', '.join(self.tx))
        return '.{ %s }' % ', '.join(parts)

    def changes_both_sides(self, previous: 'RouteConfig') -> bool:
        return self.rx != previous.rx and self.tx != previous.tx


def logical_positions(configs: list[RouteConfig]) -> tuple[list[RouteConfig], bool]:
    """
    Reduces a router's configuration sequence to one period, reporting whether it repeats.

    A PE that alternates between sending and receiving on one channel -- a halo exchange reusing a
    channel across phases, for instance -- produces ``A, B, A, B``. Storing all four would exhaust
    the router; storing ``A, B`` and letting the switch wrap around from the last position back to
    the base one expresses the same thing, which is what ``.ring_mode`` is for.

    :param configs: The configurations in the order the router takes them.
    :return: ``(period, ring_mode)``.
    """
    count = len(configs)
    for period in range(1, count):
        if count % period:
            continue
        if all(configs[index] == configs[index % period] for index in range(count)):
            return configs[:period], True
    return configs, False


def expand_positions(configs: list[RouteConfig], ring: bool = False) -> tuple[list[RouteConfig], list[int]]:
    """
    Turns a router's logical sequence of route configurations into the switch positions it holds.

    On WSE-2 a switch position carries either an input or an output, never both: the compiler
    rejects ``.pos1 = .{ .rx = RAMP, .tx = .{EAST} }`` outright. A transition that changes both sides
    is therefore split into two positions -- first the new output while the old input is kept, then
    the new input -- and costs two switch advances instead of one. The intermediate configuration is
    a pure relay, occupied only between the two wavelets that a ``close`` emits back to back.
    Architectures that accept both sides in one position (see
    ``constants.SWITCH_POSITION_ALLOWS_BOTH``) keep the transition as a single position.

    The intermediate keeps the *old* input direction, so a switch-advance wavelet arriving from the
    same neighbour as before is still accepted once the router has taken the intermediate position;
    the second wavelet would never reach the router otherwise.

    :param configs: The logical configurations, in the order the router takes them.
    :param ring: Whether the router wraps from the last configuration back to the first, which may
                 need a trailing intermediate of its own.
    :return: ``(positions, index_of)``, where ``positions`` are the hardware switch positions and
             ``index_of[i]`` is the position that ``configs[i]`` ends up at.
    """
    if not configs:
        return [], []

    def split(previous: RouteConfig, config: RouteConfig) -> bool:
        return not constants.SWITCH_POSITION_ALLOWS_BOTH and config.changes_both_sides(previous)

    positions = [configs[0]]
    index_of = [0]
    for config in configs[1:]:
        previous = positions[-1]
        if split(previous, config):
            positions.append(RouteConfig(previous.rx, config.tx))
        positions.append(config)
        index_of.append(len(positions) - 1)

    if ring and len(configs) > 1 and split(positions[-1], configs[0]):
        positions.append(RouteConfig(positions[-1].rx, configs[0].tx))
    return positions, index_of


@dataclass
class ColorSwitchPlan:
    """
    The ordered route configurations a router cycles through for one color.

    ``positions[0]`` is the base configuration, and every further entry becomes a switch position.
    Consecutive identical configurations are collapsed by :meth:`add`, so a router that keeps the
    same configuration across an epoch boundary consumes no switch position and needs no advance.
    """
    positions: list[RouteConfig] = field(default_factory=list)
    ring_mode: bool = False

    def add(self, config: RouteConfig) -> None:
        if self.positions and self.positions[-1] == config:
            return
        self.positions.append(config)

    @property
    def cycle(self) -> tuple[list[RouteConfig], bool]:
        """
        The router's configurations reduced to one period, and whether the switch wraps around.
        """
        configs, ring = logical_positions(self.positions)
        return configs, ring or self.ring_mode

    @property
    def hardware_positions(self) -> list[RouteConfig]:
        """
        The switch positions the router actually holds, with both-sided transitions split in two.
        """
        configs, ring = self.cycle
        return expand_positions(configs, ring)[0]

    @property
    def uses_switches(self) -> bool:
        return len(self.cycle[0]) > 1

    def as_csl(self) -> str:
        base = self.positions[0].as_csl()
        if not self.uses_switches:
            return '.{ .routes = %s }' % base

        hardware = self.hardware_positions
        switches = [
            '.pos%d = %s' % (index, config.as_switch_position(hardware[index - 1]))
            for index, config in enumerate(hardware[1:], start=1)
        ]
        if self.cycle[1]:
            switches.append('.ring_mode = true')
        return '.{ .routes = %s, .switches = .{ %s } }' % (base, ', '.join(switches))

    def validate(self, color: int, location: str) -> None:
        """
        Raises a ``SyntaxError`` if the plan exceeds what a router can hold.

        :param color: The color the plan is for, used for the diagnostic and for the WSE-3 check.
        :param location: A human-readable description of the PE the plan belongs to.
        """
        hardware = self.hardware_positions
        if len(hardware) > constants.SWITCH_POSITIONS:
            extra = ''
            if len(hardware) > len(self.positions):
                extra = (f' ({len(self.positions)} route configurations, {len(hardware) - len(self.positions)} '
                         'of which change both the input and the output direction and so take two '
                         'positions each)')
            raise SyntaxError(
                f'Color {color} at {location} requires {len(hardware)} switch positions{extra}, '
                f'but a router holds at most {constants.SWITCH_POSITIONS} per color on '
                f'{constants.ARCH}.\n'
                '  note: assign a different channel to some of the streams, at the cost of an '
                'additional color')

        if self.uses_switches and color not in constants.SWITCHABLE_COLORS:
            raise SyntaxError(
                f'Color {color} at {location} needs router switches, but {constants.ARCH} only '
                f'supports switches on colors {constants.SWITCHABLE_COLORS}.\n'
                '  note: assign the channel to a switchable color')


def set_color_config(x: str, y: str, color: str, plan: ColorSwitchPlan, indent: str = '') -> str:
    """
    Renders a single ``@set_color_config`` call. This is the only place that produces such text.

    :param x: The PE x coordinate expression (e.g. ``pe_x`` or ``pe_x + -1``).
    :param y: The PE y coordinate expression.
    :param color: The color expression (e.g. ``@get_color(0)``).
    :param plan: The route configurations the router cycles through.
    :param indent: Indentation to prefix the line with.
    """
    return indent + '@set_color_config(%s, %s, %s, %s);\n' % (x, y, color, plan.as_csl())


def switch_advance_payload() -> str:
    """
    Returns the ``<control>`` expression for a switch-advance control wavelet.

    The wavelet carries a single command, and every switch-configured router it passes through
    applies it: the hardware does not index the command array by hop, so a wavelet cannot advance
    one router while leaving another on the path where it is. ``ce_ignore`` keeps the wavelet from
    reaching any compute element, so no task fires when it is delivered.
    """
    return 'ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0)'


def switch_advance_statements(dsd_name: str, advances: int) -> str:
    """
    Returns the statements that retire a route configuration by advancing switches ``advances`` times.

    A transition that changes both the input and the output direction of a router occupies two
    switch positions (see :func:`expand_positions`), and therefore needs two wavelets sent back to
    back; the router is a pure relay in between.

    :param dsd_name: The fabric output descriptor the wavelets are sent through.
    :param advances: How many switch positions the routers on the path move forward.
    """
    if advances <= 0:
        raise ValueError(f'A switch advance must move at least one position, got {advances}')
    line = '@mov32(%s, %s);' % (dsd_name, switch_advance_payload())
    return '\n'.join(line for _ in range(advances))


def declare_switch_advances(rect: Rectangle[PEBlock], header: StringIO, color_map: dict[str, int],
                            dsds: cslstruct.UniqueDSDDict) -> None:
    """
    Declares the fabric output descriptors that carry a stream's switch-advance control wavelet.

    The wavelet itself is emitted by ``statements.generate_csl_statement`` from the close's
    ``switch_advance`` field; this only has to provide the descriptor it is sent through.

    The descriptor reuses the *same* output queue as the stream's data, which is mandatory rather
    than tidy: a queue is bound to one color on WSE-3 (see ``_declare_queue_initialization``), and
    queues are handed out per channel, so sending a control wavelet for one color through the queue
    that belongs to another silently fails to advance anything. A close only ever runs on a PE that
    sends the stream, so the outgoing descriptor always exists.

    :param dsds: The descriptors collected for this rectangle, to take the stream's queue from.
    """
    kept = [
        statement for statement in rect.metadata.compute.statements
        if isinstance(statement, spir.CloseStatement) and statement.switch_advance
    ]
    if not kept:
        return

    header.write('\nconst ctrl = @import_module("<control>");\n')
    for statement in kept:
        stream = stream_lifetime.underlying_stream(statement.stream_name)
        name = cslstmt.name_to_csl(stream)
        dsd_name = f'{name}_switch_dsd'
        if f'const {dsd_name}' in header.getvalue():
            continue
        queue = None
        for _, dsd in dsds.get(stream.as_ir(), ()):
            if isinstance(dsd, cslstruct.FabricDSD) and dsd.dsd_type == cslstruct.DSDType.fabout:
                queue = dsd.queue
                break
        if queue is None:
            queue = constants.OUTPUT_QUEUE_IDS[0]
        header.write(f'const {dsd_name} = @get_dsd(fabout_dsd, .{{ .extent = 1, '
                     f'.fabric_color = @get_color({color_map[name + "_OUT"]}), .control = true, '
                     f'.output_queue = @get_output_queue({queue}) }});\n')


def route_dir(dx: int, dy: int):
    """
    Helper function that returns directions for routing: (source, target).
    """
    assert abs(dx + dy) == 1
    if dx == -1:
        return ('EAST', 'WEST')
    elif dx == 1:
        return ('WEST', 'EAST')
    elif dy == -1:
        return ('SOUTH', 'NORTH')
    elif dy == 1:
        return ('NORTH', 'SOUTH')


@dataclass(frozen=True)
class _RouteSite:
    """
    One ``@set_color_config`` target: the rectangle of PEs (already shifted by the relay offset) that
    receive a route configuration for one color.
    """
    color: int
    x_range: tuple[int, int, int]
    y_range: tuple[int, int, int]

    def as_rectangle(self) -> Rectangle:
        return Rectangle(self.x_range, self.y_range, None)

    def describe(self) -> str:
        return (f'PEs [{self.x_range[0]}:{self.x_range[1]}, {self.y_range[0]}:{self.y_range[1]}]')


@dataclass
class _RouteEntry:
    """
    One route configuration contributed to a site, with the key that orders it against the other
    configurations of the same site.
    """
    config: RouteConfig
    #: ``(phase index, barrier index, statement index)``. The phase index is kernel-wide, which is
    #: what makes entries from different rectangles -- notably a relay configuration contributed by
    #: the sending rectangle -- comparable with each other. The other two come from
    #: :func:`_stream_use_order` and only separate uses *within* one rectangle, which is what
    #: sequences a receive before the send that forwards it.
    order: tuple[int, int, int]
    origin_rect: int
    origin_offset: tuple[int, int]
    stream: spir.Identifier
    #: Routing identity of the stream; stable across the per-rectangle renaming of ``inline_phases``
    group: str = ''


def _stream_use_order(compute: spir.ComputeBlock) -> dict[spir.Identifier, dict[str, tuple[int, int]]]:
    """
    Returns, per stream, the order key of its first receive and of its first send in a compute block.

    The key is ``(barrier index, statement index)``: statements are ordered first by how many phase
    barriers precede them, then by their position in the block. This is what sequences the route
    configurations of a router into switch positions.
    """
    result: dict[spir.Identifier, dict[str, tuple[int, int]]] = {}
    barrier = 0
    for index, statement in enumerate(compute.statements):
        if isinstance(statement, spir.AwaitAllStatement):
            barrier += 1
            continue
        for kind, expression in stream_lifetime.stream_references(statement):
            if kind == 'close':
                continue
            name = stream_lifetime.underlying_stream(expression)
            orders = result.setdefault(name, {})
            orders.setdefault(kind, (barrier, index))
    return result


def _offset_expression(axis: str, offset: int) -> str:
    return axis if offset == 0 else f'{axis} + {offset}'


def collect_routes(rectangles: list[Rectangle[PEBlock]],
                   color_maps: list[dict[str, int]],
                   disable_switching: bool = False) -> dict[tuple[int, int], str]:
    """
    Creates a parametric version of the Routing Graph (see the Spatial IR specification for more information) and
    returns a dictionary of code segements to add to the layout CSL file based on the streams.

    Route configurations are collected per *site* -- a rectangle of PEs and a color -- and merged
    across rectangles, because a multi-hop stream configures its relay PEs from the sending
    rectangle's loop. A site that ends up with more than one configuration is lowered to router
    switch positions, ordered by the local order of the statements that use the streams.

    :param rectangles: All rectangles involved in this kernel.
    :param color_maps: Per-rectangle mapping of stream names to colors.
    :param disable_switching: If True, emit each configuration as its own ``@set_color_config``
                              instead of merging them into switch positions.
    :return: A dictionary mapping the starting point of each rectangle to a string representing the layout instructions.
    """
    INDENT = 12 * ' '

    entries: dict[_RouteSite, list[_RouteEntry]] = {}
    for rect_index, (rect, color_map) in enumerate(zip(rectangles, color_maps)):
        for site, entry in _rectangle_route_entries(rect_index, rect, color_map):
            entries.setdefault(site, []).append(entry)

    _check_site_overlap(entries)

    result = {(rect.x_range[0], rect.y_range[0]): '' for rect in rectangles}
    for site, site_entries in entries.items():
        site_entries.sort(key=lambda entry: entry.order)

        # The site is configured from the loop of one rectangle: the one that owns these PEs if
        # there is one, otherwise the first relay that reaches them.
        owner = min(site_entries, key=lambda entry: (entry.origin_offset != (0, 0), entry.origin_rect))
        owner_rect = rectangles[owner.origin_rect]
        key = (owner_rect.x_range[0], owner_rect.y_range[0])
        x = _offset_expression('pe_x', owner.origin_offset[0])
        y = _offset_expression('pe_y', owner.origin_offset[1])
        color = f'@get_color({site.color})'

        if disable_switching:
            for entry in site_entries:
                plan = ColorSwitchPlan([entry.config])
                text = set_color_config(x, y, color, plan, INDENT)
                if text not in result[key]:
                    result[key] += text
            continue

        plan = ColorSwitchPlan()
        for entry in site_entries:
            plan.add(entry.config)
        plan.validate(site.color, site.describe())
        result[key] += set_color_config(x, y, color, plan, INDENT)

    return result


def _route_sites(rectangles: list[Rectangle[PEBlock]],
                 color_maps: list[dict[str, int]]) -> dict['_RouteSite', list['_RouteEntry']]:
    """
    Collects the route configurations of every site, sorted into switch-position order.
    """
    entries: dict[_RouteSite, list[_RouteEntry]] = {}
    for rect_index, (rect, color_map) in enumerate(zip(rectangles, color_maps)):
        for site, entry in _rectangle_route_entries(rect_index, rect, color_map):
            entries.setdefault(site, []).append(entry)
    for site_entries in entries.values():
        site_entries.sort(key=lambda entry: entry.order)
    return entries


def _channel_color_maps(rectangles: list[Rectangle[PEBlock]]) -> list[dict[str, int]]:
    """
    Builds color maps that use the stream's *channel* in place of its color.

    Switch planning has to run before colors are allocated per rectangle, and the shape of a
    router's configuration sequence only depends on the channel (a channel maps to exactly one
    color).
    """
    color_maps = []
    for rect in rectangles:
        color_map = {}
        for declaration in rect.metadata.dataflow.statements:
            if declaration.stream.routing is None:
                continue
            channel = declaration.stream.routing.resolved_channel
            if channel == 'auto':
                continue
            name = cslstmt.name_to_csl(declaration.stream_name)
            color_map[name + '_IN'] = channel
            color_map[name + '_OUT'] = channel
        color_maps.append(color_map)
    return color_maps


def plan_switch_advances(rectangles: list[Rectangle[PEBlock]]) -> int:
    """
    Determines, for every ``close`` statement, how many switch advances it has to emit.

    A close only produces code on a PE that *sends* the stream: the control wavelets it emits travel
    the path being retired. Every switch-configured router such a wavelet reaches advances -- the
    hardware applies the wavelet's single command at each of them rather than indexing a per-router
    command array -- so a close cannot move one router while leaving another on its path behind.
    All routers on the path that hold switch positions must therefore advance by the same amount,
    and that amount is how many wavelets are sent. A close on a receiving PE emits nothing; its
    router is advanced by the sending PE's wavelets.

    The result is recorded on each ``CloseStatement`` as its ``switch_advance`` field; a close that
    needs no advance keeps ``None`` there and generates no code.

    :param rectangles: The consolidated PE rectangles of the kernel, annotated in place.
    :return: The number of closes that retire a route configuration.
    :raises SyntaxError: If the routers along one path would have to advance by different amounts.
    """
    sites = _route_sites(rectangles, _channel_color_maps(rectangles))

    # Per site, where each contributed configuration sits in the router's logical sequence, and the
    # hardware switch position each of those configurations maps to.
    position_of: dict[_RouteSite, list[tuple[_RouteEntry, int]]] = {}
    hardware_index: dict[_RouteSite, list[int]] = {}
    wraps: dict[_RouteSite, tuple[bool, int]] = {}
    for site, site_entries in sites.items():
        configs: list[RouteConfig] = []
        placed = []
        for entry in site_entries:
            if not configs or configs[-1] != entry.config:
                configs.append(entry.config)
            placed.append((entry, len(configs) - 1))
        position_of[site] = placed
        cycle, ring = logical_positions(configs)
        positions, indices = expand_positions(cycle, ring)
        if len(positions) > constants.SWITCH_POSITIONS:
            continue  # Over capacity: ``collect_routes`` reports it, planning advances is moot
        # A router that wraps around returns to its base position, so every configuration has a
        # successor; otherwise the last one is final and never advances again.
        hardware_index[site] = indices
        wraps[site] = (ring, len(positions))

    planned = 0
    for rect in rectangles:
        declarations = {d.stream_name: d for d in rect.metadata.dataflow.statements}
        uses = stream_lifetime.collect_stream_uses(rect.metadata.compute)
        for statement in rect.metadata.compute.statements:
            if not isinstance(statement, spir.CloseStatement):
                continue
            statement.switch_advance = None
            name = stream_lifetime.underlying_stream(statement.stream_name)
            declaration = declarations.get(name)
            if declaration is None or name not in uses or not uses[name].sent:
                continue  # Not sent here: the sending PE's wavelets advance this router
            channel = declaration.stream.routing.resolved_channel if declaration.stream.routing else 'auto'
            offsets = stream_lifetime._stream_path_offsets(declaration)
            if channel == 'auto' or offsets is None:
                continue
            group = stream_lifetime.stream_group_key(declaration)

            # How far each switch-configured router on the path has to move, keyed by the router so
            # that a disagreement can name it.
            advances: dict[str, int] = {}
            for dx, dy in offsets:
                site = _find_site(position_of, channel,
                                  (rect.x_range[0] + dx, rect.x_range[1] + dx, rect.x_range[2]),
                                  (rect.y_range[0] + dy, rect.y_range[1] + dy, rect.y_range[2]))
                if site is None:
                    continue
                indices = hardware_index.get(site)
                if indices is None or len(indices) < 2:
                    continue  # One configuration only: no switch positions, nothing to advance
                position = _traffic_position(position_of[site], group, is_sender=(dx == 0 and dy == 0))
                if position is None:
                    continue
                ring, total = wraps[site]
                position %= len(indices)
                if position + 1 < len(indices):
                    advances[site.describe()] = indices[position + 1] - indices[position]
                elif ring:
                    advances[site.describe()] = total - indices[position]
                # Otherwise this router is on its last configuration for this color and never routes
                # anything again, so wavelets passing through may over-advance it harmlessly.

            distinct = set(advances.values())
            if not distinct:
                continue
            if len(distinct) > 1:
                detail = ', '.join(f'{where} by {count}' for where, count in sorted(advances.items()))
                raise SyntaxError(
                    f"Closing '{name.as_ir()}' has to advance the routers along its path by "
                    f'different amounts ({detail}), but one switch-advance wavelet moves every '
                    f'switch-configured router it reaches by one position.\n'
                    '  note: a control wavelet carries a single command that each router applies; '
                    'it cannot skip a router on its path\n'
                    '  note: give the streams that disagree separate channels, at the cost of an '
                    'additional color')

            statement.switch_advance = distinct.pop()
            planned += 1

    return planned


def _find_site(sites, color: int, x_range: tuple[int, int, int], y_range: tuple[int, int, int]):
    """
    Finds the site that configures a shifted rectangle of PEs for a color.

    An exact match is the common case, but a stream whose sender and receiver live in the *same*
    rectangle shifts that rectangle onto itself: PE ``(i, j)`` sends to ``(i, j+1)``, which the same
    parametric loop configures. Those lookups are resolved by intersection.
    """
    exact = _RouteSite(color=color, x_range=x_range, y_range=y_range)
    if exact in sites:
        return exact
    shifted = Rectangle(x_range, y_range, None)
    for site in sites:
        if site.color == color and site.as_rectangle().intersects(shifted):
            return site
    return None


def _traffic_position(placed: list[tuple['_RouteEntry', int]], group: str, is_sender: bool):
    """
    Returns the switch position that a stream's traffic occupies at one router.

    The sending PE injects from its own ramp, so its configuration is the one with ``rx = RAMP``;
    every router further along the path forwards traffic that arrives from the fabric. Picking the
    right one matters when a PE both receives and sends the same stream, as in a systolic chain:
    the message that retires the incoming configuration has to advance that router onto the
    outgoing one.
    """
    for entry, position in placed:
        if entry.group != group:
            continue
        if is_sender == (entry.config.rx == ('RAMP', )):
            return position
    return None


def _check_site_overlap(entries: dict['_RouteSite', list['_RouteEntry']]) -> None:
    """
    Raises a ``SyntaxError`` if two sites of the same color cover overlapping but different sets of
    PEs, which cannot be expressed as a single parametric ``@set_color_config`` loop.
    """
    sites = list(entries)
    for index, first in enumerate(sites):
        for second in sites[index + 1:]:
            if first.color != second.color:
                continue
            if not first.as_rectangle().intersects(second.as_rectangle()):
                continue
            raise SyntaxError(
                f'Color {first.color} is configured differently on overlapping but distinct PE '
                f'regions {first.describe()} and {second.describe()}.\n'
                '  note: the two regions would need separate switch sequences; split the compute '
                'blocks so that the regions coincide or are disjoint')


def _rectangle_route_entries(rect_index: int, rect: Rectangle[PEBlock],
                             color_map: dict[str, int]) -> list[tuple['_RouteSite', '_RouteEntry']]:
    """
    Collects every route configuration a single rectangle contributes, as ``(site, entry)`` pairs.
    """
    # Test whether a receive/send statement are called for creating inbound/outbound routes
    sends_recvs = analysis.sends_and_receives(rect.metadata.compute)
    use_order = _stream_use_order(rect.metadata.compute)
    collected: list[tuple[_RouteSite, _RouteEntry]] = []

    def add(offset: tuple[int, int], color: int, rx: tuple[str, ...], tx: tuple[str, ...],
            order: tuple[int, int, int], stream_name: spir.Identifier, group: str) -> None:
        site = _RouteSite(
            color=color,
            x_range=(rect.x_range[0] + offset[0], rect.x_range[1] + offset[0], rect.x_range[2]),
            y_range=(rect.y_range[0] + offset[1], rect.y_range[1] + offset[1], rect.y_range[2]))
        collected.append((site,
                          _RouteEntry(RouteConfig(rx, tx), order, rect_index, offset, stream_name, group)))

    # For each hop, make a color WEST-EAST/NORTH-SOUTH pair. For the first and last hop, pair with RAMP
    for stream in rect.metadata.dataflow.statements:
        if stream.stream_name not in sends_recvs:  # Skip unused streams
            continue
        sent, received = sends_recvs[stream.stream_name]
        group = stream_lifetime.stream_group_key(stream)
        orders = use_order.get(stream.stream_name, {})
        # A stream's epoch is fixed kernel-wide by the phase it is declared in; within that phase
        # the local statement order separates a receive from a send of the same stream, which is
        # what sequences a systolic forward.
        epoch = stream.phase if stream.phase is not None else 0
        receive_order = (epoch, ) + orders.get('receive', (0, 0))
        send_order = (epoch, ) + orders.get('send', (0, 0))
        if received:
            color_inbound = color_map[cslstmt.name_to_csl(stream.stream_name) + "_IN"]
        if sent:
            color_outbound = color_map[cslstmt.name_to_csl(stream.stream_name) + "_OUT"]

        if isinstance(stream.stream, spir.ExternStreamDeclaration):
            continue  # Extern streams do not have on-chip routing

        if isinstance(stream.stream, spir.MulticastRangeStreamDeclaration):
            if sent and received:
                raise ValueError(
                    f"Multicast stream '{stream.stream_name.as_ir()}' is both sent and received "
                    f"within the same compute rectangle [{rect.x_range[0]}:{rect.x_range[1]}, "
                    f"{rect.y_range[0]}:{rect.y_range[1]}]. "
                    "Sender and receiver compute blocks must be in separate rectangles for multicast streams.")
            if not sent:
                # All multicast routing is emitted by the rectangle that sends this stream.
                continue
            rng = stream.stream.multicast_range
            start = int(rng.start.eval())
            stop = int(rng.stop.eval())
            axis = stream.stream.multicast_axis
            is_negative = start < 0

            if axis == 'y':
                tx_dir, rx_dir = ('NORTH', 'SOUTH') if is_negative else ('SOUTH', 'NORTH')

                def _coord(k):  # noqa: E731
                    return (0, k)
            else:
                tx_dir, rx_dir = ('WEST', 'EAST') if is_negative else ('EAST', 'WEST')

                def _coord(k):  # noqa: E731
                    return (k, 0)

            # Sender: inject into fabric toward receivers.
            add((0, 0), color_outbound, ('RAMP', ), (tx_dir, ), send_order, stream.stream_name, group)

            if is_negative:
                # Negative multicast: receivers at start, start-1, …, stop+1 (stop exclusive).
                k_last = stop + 1  # farthest receiver
                gap = range(-1, start, -1)
                intermediate = range(start, k_last, -1)
            else:
                # Positive multicast: receivers at start, start+1, …, stop-1 (stop exclusive).
                k_last = stop - 1
                gap = range(1, start)
                intermediate = range(start, stop - 1)

            # Gap relay-only PEs between the sender and the first receiver.
            for k in gap:
                add(_coord(k), color_outbound, (rx_dir, ), (tx_dir, ), send_order, stream.stream_name, group)

            # Intermediate receivers: forward toward the farthest one and deliver to RAMP.
            for k in intermediate:
                add(_coord(k), color_outbound, (rx_dir, ), (tx_dir, 'RAMP'), send_order, stream.stream_name, group)

            # Last (farthest) receiver: deliver to RAMP only, no forwarding.
            add(_coord(k_last), color_outbound, (rx_dir, ), ('RAMP', ), send_order, stream.stream_name, group)
            continue

        if len(stream.stream.routing.hops) == 1:  # Inbound and outbound generated together
            route = route_dir(*stream.stream.routing.hops[0].offset)
            if sent:
                add((0, 0), color_outbound, ('RAMP', ), (route[1], ), send_order, stream.stream_name, group)
            if received:
                add((0, 0), color_inbound, (route[0], ), ('RAMP', ), receive_order, stream.stream_name, group)
        else:  # Multi-hop
            if sent:
                first_hop = stream.stream.routing.hops[0]
                add((0, 0), color_outbound, ('RAMP', ), (route_dir(*first_hop.offset)[1], ), send_order,
                    stream.stream_name, group)
                cur_offx = 0
                cur_offy = 0
                for hop in stream.stream.routing.hops[1:]:
                    route = route_dir(*hop.offset)
                    cur_offx += hop.offset[0]
                    cur_offy += hop.offset[1]
                    add((cur_offx, cur_offy), color_outbound, (route[0], ), (route[1], ), send_order,
                        stream.stream_name, group)
            if received:
                # The receiver only configures itself. Intermediate PEs are configured by the sender
                # block above, which walks forward through hops[1:] relative to the sender PE.
                last_hop = stream.stream.routing.hops[-1]
                add((0, 0), color_inbound, (route_dir(*last_hop.offset)[0], ), ('RAMP', ), receive_order,
                    stream.stream_name, group)

    return collected
