from networkx.algorithms import isomorphism

def graph_isomorphism_check(query_graphs, target_graphs, use_graph_id=True):
    """
    Compare a list of query graphs against target graphs for (sub)graph isomorphism.
    
    Args:
        query_graphs (list): List of query graphs (NetworkX graphs)
        target_graphs (list or dict): List or dict of target graphs
        use_graph_id (bool): If True, keys in target_graphs are taken from graph.graph['id'] 
                             when target_graphs is a list.
    
    Returns:
        Iso_list (list): Graphs that are isomorphic
        Iso_dict (dict): Mapping from ID/key to graph for isomorphic graphs
        notIso_list (list): Graphs that are not isomorphic
        notIso_dict (dict): Mapping from ID/key to graph for non-isomorphic graphs
    """
    Iso_list = []
    Iso_dict = {}
    notIso_list = []
    notIso_dict = {}
    
    # Convert dict to list if needed
    if isinstance(target_graphs, dict):
        targets = list(target_graphs.values())
        keys = list(target_graphs.keys())
    else:
        targets = target_graphs
        keys = [g.graph['id'] for g in targets] if use_graph_id else list(range(len(targets)))

    remaining_graphs = list(zip(keys, targets))
    
    for query in query_graphs:
        temp_remaining = []
        for k, g in remaining_graphs:
            g1, g2 = g, query
            if g1.number_of_nodes() < g2.number_of_nodes():
                g1, g2 = g2, g1

            GM = isomorphism.GraphMatcher(g1, g2)
            if GM.is_isomorphic():
                Iso_list.append(g)
                Iso_dict[k] = g
            else:
                temp_remaining.append((k, g))
        
        remaining_graphs = temp_remaining  # update remaining graphs for next query

    # Separate remaining graphs into lists/dicts
    notIso_list = [g for k, g in remaining_graphs]
    notIso_dict = {k: g for k, g in remaining_graphs}
    
    return Iso_list, Iso_dict, notIso_list, notIso_dict
