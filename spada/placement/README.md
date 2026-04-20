# Using the Placement Module


## Creating the Graph

The first step is to create a graph that models the dependencies between the
operations in the computation. This graph is a directed acyclic graph (DAG) where
the nodes are the fields and the edges represent the dependencies between
the fields. The graph is created using the `StencilGraph` class:

```python 
    # Example: Laplacian stencil
    # Define the vertices

    field_names = ['in_field', 'lap_field']

    # Define the versions of the fields

    field_versions = [0, 0]

    # Define the stencils
    stencils = [
        # in_field -> lap_field
        Stencil(np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int32), StencilDirection.PARALLEL)
    ]

    # Define the edge(s)
    # Must match in length with the stencil(s)
    # The indices are the indices of the stencils in the stencil names list
    edges = [
        (0, 1)
    ]

    # Define the domains of the fields
    # A domain has lower and upper bounds for each dimension
    # this domain has lower bound [0, 0, 0] and upper bound [256, 256, 80]
    # the dimensions are in order x, y, z
    domain = FieldDomain(np.array([[0, 0, 0], [256, 256, 80]], dtype=np.int32))
    field_domains = [domain, domain]

    # Create the graph
    graph = StencilGraph(edges, domain, field_domains, field_names, field_versions, stencils)

```


## Creating a placement

The next step is to create a placement that maps the operations in the graph to
the PEs, represented with the `Placement` class. The placement is created using
the `optimizer.best_of_k_placement`:

```python
    # Number of Tries of the Optimizer (higher k gives better results, but is slower)
    k = 10
    # Decide on shape
    shape = [2, 2]
    # Create the placed graph
    placed_graph, cost = optimizer.best_of_k_placement(graph, k, shape)

    # Get the placement
    placement = placed_graph.placement()
```

To colocate all fields, one can choose the shape to be [1, 1].
This ensures that all fields are placed in the same PE.

The placement is an `n x 2` matrix where `n` in the number of fields in the graph,
counting the versions multiple times. The placement ensures
that fields that are just different versions are always co-located.
Each row in the matrix represents the PE where the operation is placed. The first
column is the PE row and the second column is the PE column.

For example:
    
```
[[0, 0],
[0, 1],
[1, 0],
[1, 1]]
```
represents a 2x2 placement, where the first field is placed in PE (0, 0),
the second operation is placed in PE (0, 1), and so on.
