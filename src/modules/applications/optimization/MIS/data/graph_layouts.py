#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import math
import random

import networkx
import pulser

# define R_rydberg
R_rydberg = 9.75

def generate_hexagonal_graph(n_nodes:int, spacing:float,
                             filling_fraction:float=1.0) -> networkx.Graph:
    """
    Generate a hexagonal graph layout based on the number of atoms and spacing.

    Args:
        n (int): The number of nodes in the graph.
        spacing (float): The spacing between atoms.
        filling_fraction (float): The fraction of available places in the
        lattice to be filled with atoms. (default: 1.0)

    Returns:
        Graph: networkx Graph representing the hexagonal graph layout.
    """
    if filling_fraction > 1.0 or filling_fraction <= 0.0:
        raise ValueError(
            "The filling fraction must be in the domain of (0.0, 1.0]."
        )

    # Create a layout large enough to contain the desired number of atoms at
    # the filling fraction
    n_traps = int(n_nodes/filling_fraction)
    hexagonal_layout = pulser.register.special_layouts.TriangularLatticeLayout(
        n_traps=n_traps, spacing=spacing)

    # Fill the layout with traps
    reg = hexagonal_layout.hexagonal_register(n_traps) 
    ids = reg._ids
    coords = reg._coords
    coords = [l.tolist() for l in coords]
    traps = dict(zip(ids, coords))

    # Remove random atoms to get the desired number of atoms
    # This is needed if the filling fraction is below 1.0
    while len(traps) > n_nodes:
        atom_to_remove = random.choice(list(traps))
        traps.pop(atom_to_remove)

    # Rename the atoms
    i = 0
    node_positions = dict()
    for trap in traps.keys():
        node_positions[i] = traps[trap]
        i += 1

    # Create the graph
    hexagonal_graph = networkx.Graph()

    # Add the nodes
    for id, coord in node_positions.items():
        hexagonal_graph.add_node(id, pos=coord)
    
    # Generate the edges and add them to the graph
    edges = _generate_edges(node_positions=node_positions)
    hexagonal_graph.add_edges_from(edges)

    return hexagonal_graph

def _generate_edges(
        node_positions: dict,
        radius: float = R_rydberg,
    ) -> list[tuple]:
    """Generate edges between vertices within a given distance 'radius', which
    defaults to R_rydberg.

    Parameters
    ----------
    node_positions: dict
        A dictionary with the node ids as keys, and the node coordinates as
        value.
    radius: float
        When the distance between two nodes is smaller than this radius, an
        edge is generated between them.
    
    Returns
    -------
    edges: list[tuple]
        A list of 2-tuples. Each 2-tuple contains two different node ids and
        represents an edge between those two nodes.
    """
    edges = []
    vertex_keys = list(node_positions.keys())
    for i, vertex_key in enumerate(vertex_keys):
        for neighbor_key in vertex_keys[i+1:]:
            distance = _vertex_distance(node_positions[vertex_key],
                                        node_positions[neighbor_key])
            if distance <= radius:
                edges.append((vertex_key, neighbor_key))
    return edges

def _vertex_distance(v0: tuple, v1: tuple) -> float:
    """
    Calculates distance between two n-dimensional vertices.
    For 2 dimensions: distance = sqrt((x0-x1)**2 + (y0-y1)**2)
    """
    squared_difference = 0
    for coordinate0, coordinate1 in zip(v0, v1):
        squared_difference += (coordinate0 -coordinate1)**2
    return math.sqrt(squared_difference)
