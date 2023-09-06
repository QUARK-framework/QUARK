from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
from application.src.app.alb_plotting import plot_graph
from application.src.use_cases.assembly_line_balancing.alb_parser import read_data, get_indices, \
    split_lines_to_areas, TOKEN_PARSER_DISPATCHER, parse_content_to_salbp_1_problem
from application.src.use_cases.assembly_line_balancing.alb_typing import SALBP1
from application.src.utils import get_project_root, get_project_output

if __name__ == "__main__":
    # read file and parse to salb interface
    data_path = get_project_root() / "data" / "assembly_line_balancing"
    file_content = read_data(data_path / "example_instance_n=20.alb")
    token_to_index = get_indices(file_content, list(TOKEN_PARSER_DISPATCHER.keys()))
    content = split_lines_to_areas(file_content, token_to_index)

    salbp: SALBP1 = parse_content_to_salbp_1_problem(content)

    # log salb interface
    pp(salbp)

    # plot salb example
    image_path: Path = get_project_output() / "graph.gif"
    plot_graph(salbp, image_path)
    plt.show()
