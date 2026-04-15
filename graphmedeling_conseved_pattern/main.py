#from flask import Flask
#from flask import request
import contacts as ct
import graphprocessing as gp
import clustering as cl
import graphmining as gm
import common as cm
import logging
import sys
from pathlib import Path           
import numpy as np
import time

#app = Flask(__name__)    

PROJECT_ROOT = Path.cwd()
GSPAN_BINARY = PROJECT_ROOT / "gSpan" / "gSpan-64"
DEFAULT_PDBIDS_FILE = "test_even2.txt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_REGRESSION_BASELINE_DIR = PROJECT_ROOT / "results_wsl_check_fix"

if __name__ == '__main__': 
	#app.run(host="127.0.0.1", port=8080, debug=True)
	interactions,int_list = ct.readInteractions("interactions.csv")
	a_types,a_type_list = ct.readAtom_Types("atom_types.csv")
	typeCode = cm.TypeCode(a_type_list,int_list)
	typenames = cm.TypeMap("typenames.json")

	pdbids_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDBIDS_FILE
	results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_RESULTS_DIR
	baseline_results_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
	results_dir.mkdir(parents=True,exist_ok=True)
	if not GSPAN_BINARY.exists():
		raise FileNotFoundError(f"gSpan binary not found: {GSPAN_BINARY}")
		
	logging.basicConfig(level=logging.DEBUG)
	
	if not (results_dir/"graphs.txt").exists():

		logging.info("---Read PDB ids file---")
		pdbids, chains = cm.read_pdbid_file(pdbids_file)

		logging.info("---Read PDB files---")
		pdb_structures = cm.read_PDB_files(pdbids,directory="pdbfiles")

		logging.info("---Write PDB chain files---")
		cm.write_pdb_files(pdb_structures,chains,directory="pdbs")

		logging.info("---Calculate contacts---")
		contacts = ct.run_contacts(pdb_structures,chains,interactions,a_types)

		logging.info("---Generate graphs---")
		graphs, node_labels, edge_labels = ct.gen_graphs(contacts,typeCode,path=results_dir)

	else:

		logging.info("---Read graphs file---")
		graphs,node_labels,edge_labels = gp.read_graphs('graphs.txt',path=results_dir)
	
	#exit()	
	##################### Generate count matrix #############################

	count_matrix_filename = "count_matrix.csv"
	if not (results_dir/count_matrix_filename).exists():
		logging.info("---Generate counting matrix---")

		count_matrix = gp.genCountMatrix(
					graphs,node_labels,edge_labels,typeCode,
					filename=count_matrix_filename,path=results_dir)

	else:
		logging.info("---Load counting matrix---")
		count_matrix = np.genfromtxt((results_dir/count_matrix_filename), delimiter=',')

	##################### Clustering #############################
	if not (results_dir/"clusters.csv").exists():
		t0 = time.perf_counter()
		### Run SVD
		logging.info("---Run SVD on matrix---")

		n_components, n_clusters, k_neighbors, data_matrix = cl.find_n_clusters(count_matrix)
				 
		logging.info("---Run Clustering---")
		res_cluster = cl.spectral(data_matrix,n_clusters,k_neighbors)

		t1 = time.perf_counter()
		clustering_time = t1 - t0
		logging.info(f"Clustering total time: {clustering_time:.2f} seconds")
		
		clusters_file_name = "clusters.csv"
		
		with (results_dir/clusters_file_name).open(mode="w") as clusters_file:
			np.savetxt(clusters_file,res_cluster,fmt='%i', delimiter=",")
		
		clusters = cm.read_clusters(res_cluster,graphs)
	else:
		logging.info("---Load Clusters---")
		clusters_file_name = "clusters.csv"
		clusters = cm.read_clusters(clusters_file_name,graphs,path=results_dir)
		
	##################### Run gSpan #############################
	
	if not (results_dir/"gSpan.fp").exists():

		t0 = time.perf_counter()

		logging.info("---Run gSpan---")
		supports = [ "%.1f"%i for i in np.arange(0.7,1.09,0.1)]
		gm.gen_gSpan_entries(graphs,clusters,supports,
								node_labels,edge_labels,typeCode,path=results_dir,gSpan_path=str(GSPAN_BINARY))

		graph_results,clusters = gm.runGSpan(graphs,clusters,supports,
								node_labels,edge_labels,path=results_dir,gSpan_path=str(GSPAN_BINARY))
		#}'''

		t1 = time.perf_counter()
		fsm_time = t1 - t0
		logging.info(f"FSM (gSpan) total time: {fsm_time:.2f} seconds")
	
	logging.info("---Read gSpan results---")
	#'''{
	graph_results,supports = gm.read_gSpan_results(node_labels,edge_labels,filename="gSpan.fp",path=results_dir)
		#}'''

	############## Maximal

	if not (results_dir/"maximal.json").exists():
		logging.info("---Get maximal graphs---")
		maximal_patterns = gm.getMaximalGraphs(graph_results,path=results_dir)
	
	
	logging.info("---Map patterns to graphs---")
	maximal_patterns = gm.mapGraphs(clusters,"maximal.json",supports,path=results_dir)

	logging.info("---Summarize pattern occurrences at residue level---")
	gm.summarizePatternOccurrences(clusters,path=results_dir)

	logging.info("---Generate visualization-ready artifacts---")
	gm.exportVisualizationArtifacts(path=results_dir)

	logging.info("---Generate representative and aggregate hotspot views---")
	gm.exportRepresentativeAndAggregateViews(path=results_dir)

	
	logging.info("---Generate data output for visualization---")
	gm.jsonParse(clusters,maximal_patterns,a_types,supports,typeCode,"typenames.json",path=results_dir)

	logging.info("---Generate node-edge occurrence graph views---")
	gm.exportOccurrenceGraphViews(path=results_dir)
	
	gm.maximalCount(maximal_patterns,node_labels,edge_labels,typenames,typeCode,path=results_dir)

	if baseline_results_dir is None and DEFAULT_REGRESSION_BASELINE_DIR.exists() and DEFAULT_REGRESSION_BASELINE_DIR != results_dir:
		baseline_results_dir = DEFAULT_REGRESSION_BASELINE_DIR

	logging.info("---Run regression validation---")
	gm.validateRegression(results_dir,baseline_results_dir=baseline_results_dir)


    
	
