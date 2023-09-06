"""Functions to plot Assembly Line Balancing Problem"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydot

from application.src.use_cases.assembly_line_balancing.alb_factory import salbp_1_factory
from application.src.use_cases.assembly_line_balancing.alb_typing import SALBP1, Task
from application.src.utils import get_project_output


def create_graph(salbp: SALBP1, path_to_image: Path) -> None:
    """Create and save graph to show the given order of tasks in the SALBP-1

    :param salbp: an SALBP-1 instance
    :param path_to_image: path to save the graph as gif
    """
    graph: pydot.Dot = pydot.Dot(graph_type="digraph")

    task_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "gray",
        "brown",
        "pink",
        "lightblue",
        "lightgreen",
        "lightgray",
        "darkblue",
        "darkgreen",
        "darkgray",
        "darkred",
        "darkorange",
    ]

    # add nodes
    if salbp.has_valid_assignment:
        for station, tasks in salbp.task_assignment.items():
            for task in tasks:
                node_label = f"{task.id}\n{task.time}s"
                node = pydot.Node(
                    task.id,
                    label=node_label,
                    style="filled",
                    fillcolor=task_colors[station % len(task_colors)],
                )
                graph.add_node(node)
    else:
        for task in salbp.tasks:
            node_label = f"{task.id}\n{task.time}s"
            node = pydot.Node(task.id, label=node_label)
            graph.add_node(node)

    # add relation
    for task_i, task_j in salbp.preceding_tasks:
        edge = pydot.Edge(task_i.id, task_j.id)
        graph.add_edge(edge)

    # create diagram
    graph.write(path_to_image, format="gif")


def plot_graph(salbp: SALBP1, path_to_image: Path) -> None:
    """Create figure to show the given order of tasks in the SALBP-1

    :param salbp: an SALBP-1 instance
    :param path_to_image: path to save the graph as gif
    :return: figure
    """
    create_graph(salbp, path_to_image)
    # plot
    legend_data = {
        "Cycle Time": salbp.cycle_time,
        # "Order Strength": salbp.order_strength,
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.set_title("Task Relations", fontsize=20)
    ax.legend(legend_data)
    ax.text(
        0.5,
        -0.1,
        "\n".join([f"{name}: {value}" for name, value in legend_data.items()]),
        fontsize=12,
        ha="center",
        transform=ax.transAxes,
    )
    img = mpimg.imread(path_to_image)
    ax.imshow(img)


if __name__ == "__main__":
    task_1 = Task(1, 683)
    task_2 = Task(2, 461)
    task_3 = Task(3, 100)
    task_4 = Task(4, 20)
    salbp_1: SALBP1 = salbp_1_factory(
        cycle_time=100,
        tasks=[task_1, task_2, task_3, task_4],
        precedence_relations=[(task_1, task_2), (task_2, task_3), (task_1, task_4)],
    )

    image_path: Path = get_project_output() / "graph.gif"
    plot_graph(salbp_1, image_path)
    plt.show()
