import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph_operations import calculate_bond_energy, calculate_angle_energy, optimise_graph_positions

def graph_to_tikz(graph, filename: str):
    pos = nx.get_node_attributes(graph, "pos")
    
    with open(filename, "w") as fi:
    
        fi.write(r"\begin{tikzpicture}")
        for u, v in graph.edges():
            print("EDGES", u, v)
            fi.write("\t"+r"\draw [ultra thick] " + f"({pos[u][0]}, {pos[u][1]})" + r" --  " + f"({pos[v][0]}, {pos[v][1]}); % {u}-{v}\n")
            
        vels = nx.get_node_attributes(graph, "vel")
        if vels:
                fi.write("\n\n%%%% ARROWS %%%%%\n\n")
                for u in graph.nodes():
                    fi.write("\t"+r"\draw [ultra thick, ->] " + f"({pos[u][0]}, {pos[u][1]})" + r" --  " + f"({pos[u][0] + vels[u][0]}, {pos[u][1] + vels[u][1]});\n")
        fi.write(r"\end{tikzpicture}")

def main():
    G = nx.Graph()
    G.add_edges_from([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
    [2, 6], [6, 7], [7, 8], [8, 9], [9, 3],
    [9, 10], [10, 11], [11, 12], [12, 13], [13, 8],
    [4, 15], [15, 14], [14, 10]])
    
    pos = nx.spring_layout(G)
    nx.set_node_attributes(G, pos, name="pos")
    
    
    bond_energy, bond_forces = calculate_bond_energy(np.array([pos[u] for u in sorted(G.nodes())]), G, None)
    angle_energy, angle_forces = calculate_angle_energy(np.array([pos[u] for u in sorted(G.nodes())]), G, None)
    
    
    graph_to_tikz(G, "four-hexagons.tex")
    
    G.remove_edges_from([[8, 9], [3, 4]])
    G.add_edges_from([[3, 8], [4, 9]])
    
    fig, ax =plt.subplots()
    nx.draw(G, pos=pos)
    plt.show()
    
    bond_energy, bond_forces = calculate_bond_energy(np.array([pos[u] for u in sorted(G.nodes())]), G, None)
    angle_energy, angle_forces = calculate_angle_energy(np.array([pos[u] for u in sorted(G.nodes())]), G, None)
    forces = bond_forces + angle_forces
    lengths = np.sqrt(np.sum(forces**2, axis=1))
    forces /= max(lengths)
    forces_dict = {i: forces[i, :] for i in G.nodes()}
    nx.set_node_attributes(G, forces_dict, "vel")
    
    graph_to_tikz(G, "four-hexagons-after-switch.tex")
    
    G = optimise_graph_positions(G, None, do_angles=True)
    pos = nx.get_node_attributes(G, "pos")
    bond_energy, bond_forces = calculate_bond_energy(np.array([pos[u] for u in sorted(G.nodes())]), G, None)
    angle_energy, angle_forces = calculate_angle_energy(np.array([pos[u] for u in sorted(G.nodes())]), G, None)
    
    forces = bond_forces + angle_forces
    forces /= max(lengths)
    forces_dict = {i: forces[i, :] for i in G.nodes()}
    nx.set_node_attributes(G, forces_dict, "vel")
    graph_to_tikz(G, "four-hexagons-after-optimisation.tex")
    
if __name__ == "__main__":
    main()
