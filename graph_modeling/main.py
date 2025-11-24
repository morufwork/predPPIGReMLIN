import contacts as ct
import graphprocessing as gp
import clustering as cl
import graphmining as gm
import common as cm
import logging

import sys

from pathlib import Path
               
import numpy as np
import os

GSPANPATH = os.getcwd() + "/gSpan/gSpan-64"
if __name__ == '__main__': 
    
	interactions,int_list = ct.readInteractions("interactions.csv")
	a_types,a_type_list = ct.readAtom_Types("atom_types.csv")
	typeCode = cm.TypeCode(a_type_list,int_list)
	typenames = cm.TypeMap("typenames.json")
    
    
	pdbids_file = 'test_even2.txt' 
	#pdbids_file = sys.argv[1]
	path = Path("gSpan") 
	#path = Path(sys.argv[2])
    
	path.mkdir(parents=True,exist_ok=True)
	path = str(path)
		
	logging.basicConfig(level=logging.DEBUG)
	
	if not (Path(path)/"graphs.txt").exists():
        
		logging.info("---Read PDB ids file---")
		pdbids, chains = cm.read_pdbid_file(pdbids_file)

		logging.info("---Read PDB files---")
		pdb_structures = cm.read_PDB_files(pdbids,directory="pdbfiles")

		logging.info("---Write PDB chain files---")
		cm.write_pdb_files(pdb_structures,chains,directory="pdbs")

		logging.info("---Calculate contacts---")
		contacts = ct.run_contacts(pdb_structures,chains,interactions,a_types)

		logging.info("---Generate graphs---")
		graphs, node_labels, edge_labels = ct.gen_graphs(contacts,typeCode,path=path)

	else:

		logging.info("---Read graphs file---")
		graphs,node_labels,edge_labels = gp.read_graphs('graphs.txt',path=path)
	
	#exit()	
	##################### Generate count matrix #############################

	count_matrix_filename = "count_matrix.csv"
	if not (Path(path)/count_matrix_filename).exists():
		logging.info("---Generate counting matrix---")

		count_matrix = gp.genCountMatrix(
					graphs,node_labels,edge_labels,typeCode,
					filename=count_matrix_filename,path=path)

	else:
		logging.info("---Load counting matrix---")
		count_matrix = np.genfromtxt((Path(path)/count_matrix_filename), delimiter=',')

	##################### Clustering #############################
	if not (Path(path)/"clusters.csv").exists():
		
		### Run SVD
		logging.info("---Run SVD on matrix---")

		n_components, n_clusters, k_neighbors, data_matrix = cl.find_n_clusters(count_matrix)
				 
		logging.info("---Run Clustering---")
		res_cluster = cl.spectral(data_matrix,n_clusters,k_neighbors)
		
		clusters_file_name = "clusters.csv"
		
		with (Path(path)/clusters_file_name).open(mode="w") as clusters_file:
			np.savetxt(clusters_file,res_cluster,fmt='%i', delimiter=",")
		
		clusters = cm.read_clusters(res_cluster,graphs)
	else:
		logging.info("---Load Clusters---")
		clusters_file_name = "clusters.csv"
		clusters = cm.read_clusters(clusters_file_name,graphs,path=path)
		
	##################### Run gSpan #############################
	
	if not (Path(path)/"gSpan.fp").exists():

		logging.info("---Run gSpan---")
		supports = [ "%.1f"%i for i in np.arange(0.5,1.09,0.1)]
		gm.gen_gSpan_entries(graphs,clusters,supports,
								node_labels,edge_labels,typeCode,path=path,gSpan_path=GSPANPATH)

		graph_results,clusters = gm.runGSpan(graphs,clusters,supports,
								node_labels,edge_labels,path=path,gSpan_path=GSPANPATH)
		#}'''
	
	
	logging.info("---Read gSpan results---")
	#'''{
	graph_results,supports = gm.read_gSpan_results(node_labels,edge_labels,filename="gSpan.fp",path=path)
		#}'''
	#print(graph_results)
	############## Maximal

	if not (Path(path)/"maximal.json").exists():
		logging.info("---Get maximal graphs---")
		maximal_patterns = gm.getMaximalGraphs(graph_results,path=path)
	
	
	logging.info("---Map patterns to graphs---")
	maximal_patterns = gm.mapGraphs(clusters,"maximal.json",supports,path=path)

	
	logging.info("---Generate data output for visualization---")
	gm.jsonParse(clusters,maximal_patterns,a_types,supports,typeCode,"typenames.json",path=path)
	
	gm.maximalCount(maximal_patterns,node_labels,edge_labels,typenames,typeCode,path=path)
	












	
	'''import networkx as nx
	type_code = typeCode
	p_edge_labels = set()
	p_node_labels = set()
	for g_patterns in maximal_patterns:
		for sup, s_group in g_patterns.items():
			for p in s_group:
				p_edge_labels |= set(nx.get_edge_attributes(p,'type').values())
				p_node_labels |= set(nx.get_node_attributes(p,'type').values())
	
	p_edge_labels = sorted(list(p_edge_labels))
	node_labels = sorted(list(p_node_labels))
	#print(p_edge_labels)
	
	i = 1
	edge_single_labels = []
	while i < max(p_edge_labels):
		if i in p_edge_labels:
			edge_single_labels.append(i)
		i = i << 1

	print(node_labels)
	print(edge_single_labels)'''
	
	# edge_single_labels = []
	# i = edge_labels[0]
	# for j in edge_labels:
	# 	if(i == j):
	# 		edge_single_labels.append(i)
	# 		i = i << 1

	# edge_single_labels = []
	# i = edge_labels[0]
	# for j in edge_labels:
	# 	if(i == j):
	# 		edge_single_labels.append(i)
	# 		i = i << 1

	#print(patterns)
	

	#print(p_edge_labels)

	#exit()
	
	'''labels = { k:v for v,k in enumerate(node_labels + edge_single_labels,2)}
	columns = len(labels)

	r_edge_single_labels = [ i for i in reversed(edge_single_labels)]

	count_matrix = ["/".join(typenames[t] for t in type_code.type(l)) for l in labels]
	#print(count_matrix)
	for i in range(len(count_matrix)):
		if i < len(node_labels):
			count_matrix[i] = "atoms" + count_matrix[i]
		else:
			count_matrix[i] = "inter" + count_matrix[i]

	count_matrix = [['group','support'] + count_matrix] 

	print(count_matrix)'''