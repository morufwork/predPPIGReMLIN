import graphprocessing as gp
import networkx as nx
import numpy as np
import json
import sys
import os
import traceback
import logging
from common import cd
from pathlib import Path
from subprocess import call
from functools import reduce

def _json_node_id(node_id):
	if isinstance(node_id, np.integer):
		return int(node_id)
	if isinstance(node_id, np.floating):
		return float(node_id)
	return node_id

def _json_edge_id(edge_id):
	n1, n2 = edge_id
	nodes = [_json_node_id(n1), _json_node_id(n2)]
	return sorted(nodes, key=lambda value: str(value))

def _sorted_node_ids(node_ids):
	return sorted((_json_node_id(node_id) for node_id in node_ids), key=lambda value: str(value))

def _node_lookup_key(node_id):
	return str(node_id)

def _graph_node_data(graph, node_id):
	if graph.has_node(node_id):
		return graph.nodes[node_id]

	node_key = _node_lookup_key(node_id)
	if graph.has_node(node_key):
		return graph.nodes[node_key]

	try:
		int_node_id = int(node_id)
	except (TypeError, ValueError):
		int_node_id = None
	if int_node_id is not None and graph.has_node(int_node_id):
		return graph.nodes[int_node_id]

	raise KeyError(node_id)

def _residue_record(node_data):
	return {
		"pdbid": node_data["pdbid"],
		"model": int(node_data["model"]),
		"chain": node_data["chain"],
		"residueName": node_data["residueName"],
		"residueNumber": int(node_data["residueNumber"]),
		"residueInsertionCode": node_data.get("residueInsertionCode", ""),
		"heteroFlag": node_data.get("heteroFlag", ""),
		"isLigand": bool(node_data["isLigand"]),
	}

def _residue_key(residue_record):
	return (
		residue_record["pdbid"],
		residue_record["model"],
		residue_record["chain"],
		residue_record["residueName"],
		residue_record["residueNumber"],
		residue_record["residueInsertionCode"],
		residue_record["heteroFlag"],
		residue_record["isLigand"],
	)

def _centroid(coordinates):
	count = len(coordinates)
	return {
		"x": sum(point["x"] for point in coordinates) / count,
		"y": sum(point["y"] for point in coordinates) / count,
		"z": sum(point["z"] for point in coordinates) / count,
	}

def _pymol_residue_selector(residue):
	resi = str(residue["residueNumber"])
	if residue.get("residueInsertionCode"):
		resi += residue["residueInsertionCode"]

	selection_parts = [
		f"chain {residue['chain']}",
		f"resi {resi}",
	]
	if residue.get("heteroFlag"):
		selection_parts.append("hetatm")
	return "(" + " and ".join(selection_parts) + ")"

def _safe_name(value):
	return str(value).replace("/", "_")

def _pattern_key(record):
	return (int(record["cluster"]), str(record["support"]), int(record["patternId"]))

def _pattern_group_sort_key(record):
	return (int(record["cluster"]), float(record["support"]), int(record["patternId"]))

def _write_pymol_script(script_path, object_name, pdb_file, source_chain, target_chain,
		source_selection, target_selection, all_selection, note_lines):
	pymol_lines = [
		f'load {json.dumps(pdb_file)}, {object_name}',
		f'hide everything, {object_name}',
		f'show cartoon, {object_name} and chain {source_chain}+{target_chain}',
		f'color palegreen, {object_name} and chain {source_chain}',
		f'color lightblue, {object_name} and chain {target_chain}',
		f'select hotspot_source, {object_name} and ({source_selection})',
		f'select hotspot_target, {object_name} and ({target_selection})',
		f'select hotspot_all, {object_name} and ({all_selection})',
		'show sticks, hotspot_all',
		'color tv_orange, hotspot_source',
		'color hotpink, hotspot_target',
		'show spheres, hotspot_all and name CA+C1*+C2*+P',
		'set sphere_scale, 0.35, hotspot_all',
		'zoom hotspot_all, 8',
		f"orient {object_name} and chain {source_chain}+{target_chain}",
		'bg_color white',
	]
	pymol_lines.extend(note_lines)
	with script_path.open(mode="w") as pymol_script:
		pymol_script.write("\n".join(pymol_lines) + "\n")

#GSPAN_PATH = '/home/cathoud/Desktop/ppigremlin/gSpan/gSpan-64'
def gen_gSpan_entries(graphs,clusters,supports,node_labels,edge_labels,type_code,path='',gSpan_path=''):
	
	gSpanFName = 'entry.gsp'

	with cd(path):

		for key,graph_list in sorted(clusters.items(),
										key=lambda x: x[0]):
			gSpanFName='%s.gsp'%key
			gp.multigraph_to_gspan(graph_list,
					node_labels,edge_labels,type_code,gspan_fname=gSpanFName)
				
def runGSpan(graphs,clusters,supports,node_labels,edge_labels,path='',gSpan_path=''):
	
	graphs_dict = clusters
	
	gSpan_out = (Path(path)/'gSpan.txt').open(mode='w') #gSpan log
	gSpan_results = dict()
	
	#Change context 
	with cd(path):

		for key,graph_list in sorted(graphs_dict.items(),
										key=lambda x: x[0]):
			gSpanFName = str(key) + ".gsp"
			
			temp_gSpan_results = []
			for min_sup in supports:
				call([gSpan_path,"-f",gSpanFName,"-s",str(min_sup),"-o","-i"],stdout=gSpan_out)
	
				Path(gSpanFName+'.fp').rename('%s_%s.fp'%(key,min_sup))

				temp_gSpan_results.append('%s_%s.fp'%(key,min_sup))

			gSpan_results[int(key)] = temp_gSpan_results

		gSpan_results = {'results':gSpan_results, 'supports':supports}
		with open('gSpan.fp','w') as out_gspan_files:
			json.dump(gSpan_results,out_gspan_files,indent=4)

	return gSpan_results,graphs_dict

def read_gSpan_results(node_labels,edge_labels,filename="gSpan.fp",path=""):

	with (Path(path)/filename).open() as filename_list:
		result_files = json.load(filename_list)

	gSpan_results = dict()
	supports = result_files['supports']
	with cd(path):		
		for cluster,files in result_files['results'].items():
			temp_results = dict()
			for filename in files:
				key = filename.split('_')[1].split('f')[0][:-1]
				temp_results[key] = gp.gspan_to_graph(filename,node_labels,edge_labels)
			
			gSpan_results[int(cluster)] = temp_results
	return gSpan_results,supports
	
def getMaximalGraphs(clusters,file="maximal.json",path=""):	

	j_maximal = []
	maximal_graphs = []
	
	for n_cluster,cluster in clusters.items():
	
		j_temp = dict()
		temp = dict()
	
		for min_sup, graphs in sorted(cluster.items()):
	
			g =	maximal(graphs)
			temp[min_sup] = [ {"graph": j["graph"], "l_graph": j["l_graph"]} for j in g ]
			j_temp[min_sup] = ["".join([i for i in nx.generate_gml(j["graph"])]) for j in g]

		maximal_graphs.append(temp)
		j_maximal.append(j_temp)
	

	with (Path(path) / file ).open(mode='w') as j_patterns_file:
		j_patterns_file.write(json.dumps(j_maximal,indent=4))

	return maximal_graphs
		
def maximal(graphs):

	############# Filter: remove 1-vertex graphs
	graphs = [i for i in filter(lambda x: x.number_of_nodes() > 1,graphs)]

	if not graphs:
		return []

	graphs.sort(key=lambda x: -x.number_of_nodes())

	### Generate Line Graphs
	graphs = [{"graph": g, "l_graph" : gp.line_graph(g)} for g in graphs]
	
	############# Get maximals

	######## Node Split
	graphs_holder = graphs
	graphs = []
	last_num_nodes =-1
	while graphs_holder:
		num_nodes = graphs_holder[-1]["graph"].number_of_nodes()
		if  num_nodes != last_num_nodes:
			graphs.append([])
			last_num_nodes = num_nodes

		graphs[-1].append(graphs_holder.pop())

	######## Edge Split	
	graphs = [sorted(i,key = lambda x: -x['graph'].number_of_edges()) for i in graphs]

	######## Graph Subgraph Isomorphism (Level 1)
	nm = lambda x,y: x['type'] >= y['type'] and (x['type'] & y['type'])
	em = nx.isomorphism.numerical_node_match(["type"],[""])

	marked = [np.full((len(g)),False) for g in graphs]
	
	for g in range(1,len(graphs)):
		count = 0

		for i in range(len(graphs[g])):
			j_size = len(graphs[g-1])
			
			if count >= j_size:
				break

			for j in range(j_size):
				if marked[g-1][j]:
					continue

				m = nx.isomorphism.GraphMatcher(graphs[g][i]["l_graph"],graphs[g-1][j]["l_graph"],
							edge_match=em,node_match=nm)

				if m.subgraph_is_isomorphic():
					count+=1		
					marked[g-1][j] = True

				if count >= j_size:
					break
			

	graphs_holder = [np.array(g, dtype=object) for g in graphs]
	graphs = [g[i] for g,i in zip(graphs_holder,np.invert(np.array(marked)))]
	# Reverse the grouped buckets without forcing NumPy to coerce ragged lists.
	graphs = np.concatenate(list(reversed(graphs)))
	
	######## Graph Subgraph Isomorphism (Level 2)
	nm = lambda x,y: x['type'] >= y['type'] and (x['type'] & y['type'])
	em = nx.isomorphism.numerical_node_match(["type"],[""])
	
	marked = np.full((len(graphs)),False)
	count = 0
	
	for i in range(len(graphs)):
	
		for j in range(len(graphs)):
			if(i == j or marked[j]):
				continue

			if(graphs[i]["graph"].number_of_nodes() >= graphs[j]["graph"].number_of_nodes()):
				
				m = nx.isomorphism.GraphMatcher(graphs[i]["l_graph"],graphs[j]["l_graph"],
							edge_match=em,node_match=nm)

				if(m.subgraph_is_isomorphic()):
					count+=1	
					marked[j] = True

	for i,g in enumerate(graphs):
		g["graph"].graph['id'] = i

	return list(graphs[np.invert(marked)])

def filter_maximal(patterns,n=10):
	path = ""
	if isinstance(patterns,str):
		try:
			with (Path(path)/patterns).open() as p_file:
				patterns = json.load(p_file)
				patterns = [{ key:[nx.parse_gml(g) for g in v] for key,v in p.items()} for p in patterns]
				r_patterns = patterns
		except IOError as e:
			raise e

	for p in patterns:
		sups = sorted(p.keys())
		
		for key in sups:
			if len(p[key]) < 1:
				continue
			
			p[key].sort(key=lambda x:-x.number_of_edges())
			p[key].sort(key=lambda x:-x.number_of_nodes())
			p[key] = p[key][:n]
			
	return patterns

def mapGraphs(clusters,patterns,supports,path=""):

	if isinstance(patterns,str):
		try:
			with (Path(path)/patterns).open() as p_file:
				patterns = json.load(p_file)
				patterns = [{ key:[nx.parse_gml(g) for g in v] for key,v in p.items()} for p in patterns]
				r_patterns = patterns
		except IOError as e:
			raise e

	patterns = filter_maximal(patterns)
	
	cluster_items = sorted(clusters.items())
	clusters = [cluster_graphs for _, cluster_graphs in cluster_items]
	pattern_occurrences = []

	for c in clusters:
		for graphs in c:
			
			for n1,n2 in graphs.edges():
				graphs[n1][n2]["patterns"] = {k:set() for k in supports}
			
			for n in graphs.nodes():
				graphs.nodes[n]["patterns"] = {k:set() for k in supports}
			
			graphs.graph["l_graph"] = gp.line_graph(graphs)

	
	# map_file = (Path(path)/"p_mappings").open(mode="a")
	

	for (cluster_id, _), p_graphs, graphs in zip(cluster_items,patterns,clusters):
		for min_sup, p_graphs in sorted(p_graphs.items(),key=lambda x: float(x[0])):
			for ip,p in enumerate(p_graphs):
				
				pl = gp.line_graph(p)
				for s in p.graph['ocur']:
					g = graphs[int(s)]
					gl = g.graph["l_graph"]

					nm = lambda x,y: x['type'] >= y['type'] and (x['type'] & y['type'])

					em = nx.isomorphism.numerical_node_match(["type"],[""])
					m = nx.isomorphism.GraphMatcher(gl,pl,edge_match=em,node_match=nm)

					m_iter = m.subgraph_isomorphisms_iter()

					for mapping in m_iter:
						matched_graph_edges = sorted(
							[_json_edge_id(edge_id) for edge_id in mapping.keys()],
							key=lambda edge_id: (str(edge_id[0]), str(edge_id[1])),
						)
						matched_pattern_edges = sorted(
							[_json_edge_id(edge_id) for edge_id in mapping.values()],
							key=lambda edge_id: (str(edge_id[0]), str(edge_id[1])),
						)
						matched_graph_nodes = _sorted_node_ids(
							{node_id for edge_id in mapping.keys() for node_id in edge_id}
						)
						matched_pattern_nodes = _sorted_node_ids(
							{node_id for edge_id in mapping.values() for node_id in edge_id}
						)

						pattern_occurrences.append({
							"cluster": int(cluster_id),
							"support": str(min_sup),
							"patternId": int(p.graph['id']),
							"graphId": int(g.graph['id']),
							"matchedGraphNodeIds": matched_graph_nodes,
							"matchedGraphEdgeIds": matched_graph_edges,
							"matchedPatternNodeIds": matched_pattern_nodes,
							"matchedPatternEdgeIds": matched_pattern_edges,
						})

						## Mapping Nodes
						for k,v in mapping.items():

							# mapped_nodes.add(k[0])
							# mapped_nodes.add(k[1])
							# mapped_edges.add("-".join(sorted(k)))
							g[k[0]][k[1]]["patterns"][min_sup].add(p.graph['id'])
							g.nodes[k[0]]['patterns'][min_sup].add(p.graph['id'])
							g.nodes[k[1]]['patterns'][min_sup].add(p.graph['id'])
	with (Path(path)/"pattern_occurrences.json").open(mode="w") as occurrences_file:
		json.dump(pattern_occurrences, occurrences_file, indent=4)

	return patterns

def summarizePatternOccurrences(clusters, occurrence_file="pattern_occurrences.json",
		residue_occurrence_file="pattern_occurrences_residue.json",
		hotspots_summary_file="pattern_hotspots_summary.csv", path=""):

	with (Path(path)/occurrence_file).open() as occurrences_handle:
		pattern_occurrences = json.load(occurrences_handle)

	graph_by_id = {}
	for _, graph_list in sorted(clusters.items()):
		for graph in graph_list:
			graph_by_id[int(graph.graph["id"])] = graph

	residue_occurrences = []
	hotspots = {}

	for occurrence_idx, occurrence in enumerate(pattern_occurrences):
		graph = graph_by_id[int(occurrence["graphId"])]
		matched_nodes = []
		residue_records = {}
		chain_pair = {
			"source": graph.graph["source"],
			"target": graph.graph["target"],
		}
		structure_context = {
			"pdbid": graph.graph["pdbid"],
			"graphId": int(graph.graph["id"]),
			"chainPair": dict(chain_pair),
			"pdbFile": str(Path("pdbfiles") / ("pdb%s.ent" % graph.graph["pdbid"].lower())),
			"chainPairFile": str(Path("pdbs") / ("%s.%s.pdb" % (graph.graph["pdbid"], graph.graph["source"]))),
		}

		for node_id in occurrence["matchedGraphNodeIds"]:
			node_data = _graph_node_data(graph, node_id)
			atom_record = {
				"nodeId": _json_node_id(node_id),
				"pdbid": node_data["pdbid"],
				"model": int(node_data["model"]),
				"chain": node_data["chain"],
				"residueName": node_data["residueName"],
				"residueNumber": int(node_data["residueNumber"]),
				"residueInsertionCode": node_data.get("residueInsertionCode", ""),
				"heteroFlag": node_data.get("heteroFlag", ""),
				"atomName": node_data["atomName"],
				"atomSerial": int(node_data["atomSerial"]),
				"isLigand": bool(node_data["isLigand"]),
				"coordinates": {
					"x": float(node_data["coordX"]),
					"y": float(node_data["coordY"]),
					"z": float(node_data["coordZ"]),
				},
			}
			matched_nodes.append(atom_record)

			residue_record = _residue_record(node_data)
			residue_id = _residue_key(residue_record)
			residue_entry = residue_records.get(residue_id)
			if residue_entry is None:
				residue_entry = dict(residue_record)
				residue_entry["matchedAtomNodeIds"] = []
				residue_entry["matchedAtomSerials"] = []
				residue_entry["matchedAtomNames"] = []
				residue_entry["matchedAtomCoordinates"] = []
				residue_records[residue_id] = residue_entry

			residue_entry["matchedAtomNodeIds"].append(atom_record["nodeId"])
			residue_entry["matchedAtomSerials"].append(atom_record["atomSerial"])
			residue_entry["matchedAtomNames"].append(atom_record["atomName"])
			residue_entry["matchedAtomCoordinates"].append(dict(atom_record["coordinates"]))

		residue_list = sorted(
			residue_records.values(),
			key=lambda item: (
				item["pdbid"], item["model"], item["chain"], item["residueNumber"],
				item["residueInsertionCode"], item["atomName"] if "atomName" in item else ""
			),
		)
		for residue_entry in residue_list:
			residue_entry["matchedAtomNodeIds"] = _sorted_node_ids(residue_entry["matchedAtomNodeIds"])
			residue_entry["matchedAtomSerials"] = sorted(set(residue_entry["matchedAtomSerials"]))
			residue_entry["matchedAtomNames"] = sorted(set(residue_entry["matchedAtomNames"]))
			residue_entry["representativeCoordinates"] = _centroid(residue_entry["matchedAtomCoordinates"])
			residue_entry["chainPair"] = dict(chain_pair)
			residue_entry["pdbFile"] = structure_context["pdbFile"]
			residue_entry["chainPairFile"] = structure_context["chainPairFile"]

		residue_occurrence = dict(occurrence)
		residue_occurrence["occurrenceId"] = occurrence_idx
		residue_occurrence["structureContext"] = structure_context
		residue_occurrence["matchedAtoms"] = matched_nodes
		residue_occurrence["matchedResidues"] = residue_list
		residue_occurrences.append(residue_occurrence)

		for residue_entry in residue_list:
			hotspot_key = (
				int(occurrence["cluster"]),
				str(occurrence["support"]),
				int(occurrence["patternId"]),
				residue_entry["pdbid"],
				residue_entry["model"],
				residue_entry["chain"],
				residue_entry["residueName"],
				residue_entry["residueNumber"],
				residue_entry["residueInsertionCode"],
				residue_entry["heteroFlag"],
				residue_entry["isLigand"],
			)
			hotspot_entry = hotspots.get(hotspot_key)
			if hotspot_entry is None:
				hotspot_entry = {
					"cluster": int(occurrence["cluster"]),
					"support": str(occurrence["support"]),
					"patternId": int(occurrence["patternId"]),
					"pdbid": residue_entry["pdbid"],
					"model": residue_entry["model"],
					"chain": residue_entry["chain"],
					"residueName": residue_entry["residueName"],
					"residueNumber": residue_entry["residueNumber"],
					"residueInsertionCode": residue_entry["residueInsertionCode"],
					"heteroFlag": residue_entry["heteroFlag"],
					"isLigand": residue_entry["isLigand"],
					"occurrenceCount": 0,
					"atomMatchCount": 0,
					"graphIds": set(),
					"occurrenceIds": set(),
					"chainPairSources": set(),
					"chainPairTargets": set(),
					"representativeCoordinates": [],
					"pdbFile": structure_context["pdbFile"],
					"chainPairFile": structure_context["chainPairFile"],
				}
				hotspots[hotspot_key] = hotspot_entry

			hotspot_entry["occurrenceCount"] += 1
			hotspot_entry["atomMatchCount"] += len(residue_entry["matchedAtomNodeIds"])
			hotspot_entry["graphIds"].add(int(occurrence["graphId"]))
			hotspot_entry["occurrenceIds"].add(occurrence_idx)
			hotspot_entry["chainPairSources"].add(chain_pair["source"])
			hotspot_entry["chainPairTargets"].add(chain_pair["target"])
			hotspot_entry["representativeCoordinates"].append(dict(residue_entry["representativeCoordinates"]))

	with (Path(path)/residue_occurrence_file).open(mode="w") as residue_occurrences_handle:
		json.dump(residue_occurrences, residue_occurrences_handle, indent=4)

	hotspot_rows = []
	for hotspot_entry in hotspots.values():
		row = dict(hotspot_entry)
		row["graphIds"] = sorted(hotspot_entry["graphIds"])
		row["occurrenceIds"] = sorted(hotspot_entry["occurrenceIds"])
		row["chainPairSources"] = sorted(hotspot_entry["chainPairSources"])
		row["chainPairTargets"] = sorted(hotspot_entry["chainPairTargets"])
		row["averageRepresentativeCoordinates"] = _centroid(hotspot_entry["representativeCoordinates"])
		row["graphCount"] = len(row["graphIds"])
		row["occurrenceIds"] = ";".join(str(value) for value in row["occurrenceIds"])
		row["graphIds"] = ";".join(str(value) for value in row["graphIds"])
		row["chainPairSources"] = ";".join(row["chainPairSources"])
		row["chainPairTargets"] = ";".join(row["chainPairTargets"])
		row["avgCoordX"] = row["averageRepresentativeCoordinates"]["x"]
		row["avgCoordY"] = row["averageRepresentativeCoordinates"]["y"]
		row["avgCoordZ"] = row["averageRepresentativeCoordinates"]["z"]
		del row["representativeCoordinates"]
		hotspot_rows.append(row)

	hotspot_rows.sort(
		key=lambda item: (
			item["cluster"], item["support"], item["patternId"], -item["occurrenceCount"],
			item["pdbid"], item["chain"], item["residueNumber"], item["residueInsertionCode"],
		)
	)

	headers = [
		"cluster", "support", "patternId", "pdbid", "model", "chain", "residueName",
		"residueNumber", "residueInsertionCode", "heteroFlag", "isLigand",
		"occurrenceCount", "atomMatchCount", "graphCount", "graphIds", "occurrenceIds",
		"chainPairSources", "chainPairTargets", "pdbFile", "chainPairFile",
		"avgCoordX", "avgCoordY", "avgCoordZ",
	]
	with (Path(path)/hotspots_summary_file).open(mode="w") as hotspots_summary_handle:
		hotspots_summary_handle.write(",".join(headers) + "\n")
		for row in hotspot_rows:
			hotspots_summary_handle.write(",".join([
				json.dumps(row[header]) if isinstance(row[header], str) else str(row[header])
				for header in headers
			]) + "\n")

	return residue_occurrences, hotspot_rows

def exportVisualizationArtifacts(
		residue_occurrence_file="pattern_occurrences_residue.json",
		visualization_dir="visualization",
		manifest_file="visualization_manifest.json",
		path=""):

	with (Path(path)/residue_occurrence_file).open() as residue_occurrences_handle:
		residue_occurrences = json.load(residue_occurrences_handle)

	visualization_root = Path(path) / visualization_dir
	pymol_dir = visualization_root / "pymol_occurrences"
	pymol_dir.mkdir(parents=True, exist_ok=True)

	manifest = []
	for occurrence in residue_occurrences:
		context = occurrence["structureContext"]
		pdb_file = str((Path(context["pdbFile"])).resolve())
		source_chain = context["chainPair"]["source"]
		target_chain = context["chainPair"]["target"]
		object_name = (
			f"occ_{occurrence['occurrenceId']}_"
			f"c{occurrence['cluster']}_p{occurrence['patternId']}_"
			f"s{_safe_name(occurrence['support'])}"
		)
		script_name = (
			f"occurrence_{occurrence['occurrenceId']}_"
			f"cluster_{occurrence['cluster']}_pattern_{occurrence['patternId']}_"
			f"support_{_safe_name(occurrence['support'])}.pml"
		)
		script_path = pymol_dir / script_name

		source_residues = [r for r in occurrence["matchedResidues"] if not r["isLigand"]]
		target_residues = [r for r in occurrence["matchedResidues"] if r["isLigand"]]
		source_selection = " or ".join(_pymol_residue_selector(r) for r in source_residues) or "none"
		target_selection = " or ".join(_pymol_residue_selector(r) for r in target_residues) or "none"
		all_selection = " or ".join(_pymol_residue_selector(r) for r in occurrence["matchedResidues"]) or "none"

		pymol_lines = [
			f'load {json.dumps(pdb_file)}, {object_name}',
			f'hide everything, {object_name}',
			f'show cartoon, {object_name} and chain {source_chain}+{target_chain}',
			f'color palegreen, {object_name} and chain {source_chain}',
			f'color lightblue, {object_name} and chain {target_chain}',
			f'select hotspot_source, {object_name} and ({source_selection})',
			f'select hotspot_target, {object_name} and ({target_selection})',
			f'select hotspot_all, {object_name} and ({all_selection})',
			'show sticks, hotspot_all',
			'color tv_orange, hotspot_source',
			'color hotpink, hotspot_target',
			'show spheres, hotspot_all and name CA+C1*+C2*+P',
			'set sphere_scale, 0.35, hotspot_all',
			'zoom hotspot_all, 8',
			f"orient {object_name} and chain {source_chain}+{target_chain}",
			f'set_name hotspot_all, hotspot_occurrence_{occurrence["occurrenceId"]}',
			f'set_name hotspot_source, hotspot_source_{occurrence["occurrenceId"]}',
			f'set_name hotspot_target, hotspot_target_{occurrence["occurrenceId"]}',
			'bg_color white',
			f'# patternId={occurrence["patternId"]} support={occurrence["support"]} graphId={occurrence["graphId"]}',
		]
		with script_path.open(mode="w") as pymol_script:
			pymol_script.write("\n".join(pymol_lines) + "\n")

		manifest.append({
			"occurrenceId": occurrence["occurrenceId"],
			"cluster": occurrence["cluster"],
			"support": occurrence["support"],
			"patternId": occurrence["patternId"],
			"graphId": occurrence["graphId"],
			"pdbid": context["pdbid"],
			"chainPair": context["chainPair"],
			"pdbFile": pdb_file,
			"chainPairFile": context["chainPairFile"],
			"script": str(script_path),
			"matchedResidues": occurrence["matchedResidues"],
		})

	with (visualization_root / manifest_file).open(mode="w") as manifest_handle:
		json.dump(manifest, manifest_handle, indent=4)

	return manifest

def exportRepresentativeAndAggregateViews(
		residue_occurrence_file="pattern_occurrences_residue.json",
		hotspots_summary_file="pattern_hotspots_summary.csv",
		visualization_dir="visualization",
		representative_manifest_file="representative_patterns.json",
		aggregate_manifest_file="aggregate_patterns.json",
		pattern_summary_file="pattern_residue_ranking.csv",
		path=""):

	with (Path(path)/residue_occurrence_file).open() as residue_occurrences_handle:
		residue_occurrences = json.load(residue_occurrences_handle)

	visualization_root = Path(path) / visualization_dir
	rep_dir = visualization_root / "pymol_representative_patterns"
	agg_dir = visualization_root / "pymol_aggregate_patterns"
	rep_dir.mkdir(parents=True, exist_ok=True)
	agg_dir.mkdir(parents=True, exist_ok=True)

	pattern_occurrences = {}
	for occurrence in residue_occurrences:
		pattern_occurrences.setdefault(_pattern_key(occurrence), []).append(occurrence)

	representative_manifest = []
	aggregate_manifest = []
	pattern_summary_rows = []

	for pattern_key, occurrences in sorted(pattern_occurrences.items(), key=lambda item: _pattern_group_sort_key({
			"cluster": item[0][0], "support": item[0][1], "patternId": item[0][2]
		})):
		cluster, support, pattern_id = pattern_key
		occurrences.sort(
			key=lambda occ: (
				-len(occ["matchedResidues"]),
				-len(occ["matchedAtoms"]),
				int(occ["graphId"]),
				int(occ["occurrenceId"]),
			)
		)
		representative = occurrences[0]
		context = representative["structureContext"]
		rep_pdb_file = str((Path(context["pdbFile"])).resolve())
		source_chain = context["chainPair"]["source"]
		target_chain = context["chainPair"]["target"]

		rep_source_residues = [r for r in representative["matchedResidues"] if not r["isLigand"]]
		rep_target_residues = [r for r in representative["matchedResidues"] if r["isLigand"]]
		rep_source_selection = " or ".join(_pymol_residue_selector(r) for r in rep_source_residues) or "none"
		rep_target_selection = " or ".join(_pymol_residue_selector(r) for r in rep_target_residues) or "none"
		rep_all_selection = " or ".join(_pymol_residue_selector(r) for r in representative["matchedResidues"]) or "none"

		rep_script_name = (
			f"cluster_{cluster}_pattern_{pattern_id}_support_{_safe_name(support)}_representative.pml"
		)
		rep_script_path = rep_dir / rep_script_name
		rep_object_name = f"rep_c{cluster}_p{pattern_id}_s{_safe_name(support)}"
		_write_pymol_script(
			rep_script_path,
			rep_object_name,
			rep_pdb_file,
			source_chain,
			target_chain,
			rep_source_selection,
			rep_target_selection,
			rep_all_selection,
			[
				f'set_name hotspot_all, representative_hotspot_{cluster}_{pattern_id}',
				f'set_name hotspot_source, representative_source_{cluster}_{pattern_id}',
				f'set_name hotspot_target, representative_target_{cluster}_{pattern_id}',
				f'# representative occurrenceId={representative["occurrenceId"]} graphId={representative["graphId"]}',
			],
		)
		representative_manifest.append({
			"cluster": cluster,
			"support": support,
			"patternId": pattern_id,
			"representativeOccurrenceId": representative["occurrenceId"],
			"graphId": representative["graphId"],
			"pdbid": context["pdbid"],
			"chainPair": context["chainPair"],
			"pdbFile": rep_pdb_file,
			"chainPairFile": context["chainPairFile"],
			"script": str(rep_script_path),
			"matchedResidues": representative["matchedResidues"],
			"occurrenceCount": len(occurrences),
		})

		aggregate_residues = {}
		for occurrence in occurrences:
			for residue in occurrence["matchedResidues"]:
				residue_key = _residue_key(residue)
				entry = aggregate_residues.get(residue_key)
				if entry is None:
					entry = {
						"cluster": cluster,
						"support": support,
						"patternId": pattern_id,
						"pdbid": residue["pdbid"],
						"model": residue["model"],
						"chain": residue["chain"],
						"residueName": residue["residueName"],
						"residueNumber": residue["residueNumber"],
						"residueInsertionCode": residue["residueInsertionCode"],
						"heteroFlag": residue["heteroFlag"],
						"isLigand": residue["isLigand"],
						"occurrenceCount": 0,
						"atomMatchCount": 0,
						"graphIds": set(),
						"occurrenceIds": set(),
						"representativeCoordinates": [],
						"pdbFile": residue["pdbFile"],
						"chainPairFile": residue["chainPairFile"],
						"chainPairSources": set(),
						"chainPairTargets": set(),
					}
					aggregate_residues[residue_key] = entry

				entry["occurrenceCount"] += 1
				entry["atomMatchCount"] += len(residue["matchedAtomNodeIds"])
				entry["graphIds"].add(int(occurrence["graphId"]))
				entry["occurrenceIds"].add(int(occurrence["occurrenceId"]))
				entry["representativeCoordinates"].append(dict(residue["representativeCoordinates"]))
				entry["chainPairSources"].add(occurrence["structureContext"]["chainPair"]["source"])
				entry["chainPairTargets"].add(occurrence["structureContext"]["chainPair"]["target"])

		aggregate_rows = []
		for entry in aggregate_residues.values():
			entry["graphCount"] = len(entry["graphIds"])
			entry["averageRepresentativeCoordinates"] = _centroid(entry["representativeCoordinates"])
			entry["graphIds"] = sorted(entry["graphIds"])
			entry["occurrenceIds"] = sorted(entry["occurrenceIds"])
			entry["chainPairSources"] = sorted(entry["chainPairSources"])
			entry["chainPairTargets"] = sorted(entry["chainPairTargets"])
			aggregate_rows.append(entry)

		aggregate_rows.sort(
			key=lambda row: (
				-row["occurrenceCount"],
				-row["atomMatchCount"],
				row["pdbid"],
				row["chain"],
				row["residueNumber"],
				row["residueInsertionCode"],
			)
		)

		agg_context = representative["structureContext"]
		agg_pdb_file = str((Path(agg_context["pdbFile"])).resolve())
		top_rows = aggregate_rows[: min(12, len(aggregate_rows))]
		visualized_rows = [
			row for row in top_rows
			if row["pdbid"] == agg_context["pdbid"]
		]
		if not visualized_rows:
			visualized_rows = [
				row for row in aggregate_rows
				if row["pdbid"] == agg_context["pdbid"]
			][: min(12, len(aggregate_rows))]
		if not visualized_rows:
			visualized_rows = [
				{
					"chain": residue["chain"],
					"residueNumber": residue["residueNumber"],
					"residueInsertionCode": residue["residueInsertionCode"],
					"heteroFlag": residue["heteroFlag"],
					"isLigand": residue["isLigand"],
				}
				for residue in representative["matchedResidues"]
			]

		agg_source_selection = " or ".join(
			_pymol_residue_selector(row) for row in visualized_rows if not row["isLigand"]
		) or "none"
		agg_target_selection = " or ".join(
			_pymol_residue_selector(row) for row in visualized_rows if row["isLigand"]
		) or "none"
		agg_all_selection = " or ".join(_pymol_residue_selector(row) for row in visualized_rows) or "none"

		agg_script_name = (
			f"cluster_{cluster}_pattern_{pattern_id}_support_{_safe_name(support)}_aggregate.pml"
		)
		agg_script_path = agg_dir / agg_script_name
		agg_object_name = f"agg_c{cluster}_p{pattern_id}_s{_safe_name(support)}"
		_write_pymol_script(
			agg_script_path,
			agg_object_name,
			agg_pdb_file,
			source_chain,
			target_chain,
			agg_source_selection,
			agg_target_selection,
			agg_all_selection,
			[
				f'set_name hotspot_all, aggregate_hotspot_{cluster}_{pattern_id}',
				f'set_name hotspot_source, aggregate_source_{cluster}_{pattern_id}',
				f'set_name hotspot_target, aggregate_target_{cluster}_{pattern_id}',
				f'# aggregate top residues for cluster={cluster} patternId={pattern_id} support={support}',
			],
		)
		aggregate_manifest.append({
			"cluster": cluster,
			"support": support,
			"patternId": pattern_id,
			"occurrenceCount": len(occurrences),
			"representativeOccurrenceId": representative["occurrenceId"],
			"pdbid": agg_context["pdbid"],
			"chainPair": agg_context["chainPair"],
			"pdbFile": agg_pdb_file,
			"chainPairFile": agg_context["chainPairFile"],
			"script": str(agg_script_path),
			"topResidues": top_rows,
			"visualizedResidues": visualized_rows,
		})

		for rank, row in enumerate(aggregate_rows, start=1):
			pattern_summary_rows.append({
				"cluster": cluster,
				"support": support,
				"patternId": pattern_id,
				"rank": rank,
				"pdbid": row["pdbid"],
				"model": row["model"],
				"chain": row["chain"],
				"residueName": row["residueName"],
				"residueNumber": row["residueNumber"],
				"residueInsertionCode": row["residueInsertionCode"],
				"heteroFlag": row["heteroFlag"],
				"isLigand": row["isLigand"],
				"occurrenceCount": row["occurrenceCount"],
				"atomMatchCount": row["atomMatchCount"],
				"graphCount": row["graphCount"],
				"graphIds": ";".join(str(value) for value in row["graphIds"]),
				"occurrenceIds": ";".join(str(value) for value in row["occurrenceIds"]),
				"chainPairSources": ";".join(row["chainPairSources"]),
				"chainPairTargets": ";".join(row["chainPairTargets"]),
				"pdbFile": row["pdbFile"],
				"chainPairFile": row["chainPairFile"],
				"avgCoordX": row["averageRepresentativeCoordinates"]["x"],
				"avgCoordY": row["averageRepresentativeCoordinates"]["y"],
				"avgCoordZ": row["averageRepresentativeCoordinates"]["z"],
			})

	with (visualization_root / representative_manifest_file).open(mode="w") as representative_handle:
		json.dump(representative_manifest, representative_handle, indent=4)
	with (visualization_root / aggregate_manifest_file).open(mode="w") as aggregate_handle:
		json.dump(aggregate_manifest, aggregate_handle, indent=4)

	headers = [
		"cluster", "support", "patternId", "rank", "pdbid", "model", "chain", "residueName",
		"residueNumber", "residueInsertionCode", "heteroFlag", "isLigand", "occurrenceCount",
		"atomMatchCount", "graphCount", "graphIds", "occurrenceIds", "chainPairSources",
		"chainPairTargets", "pdbFile", "chainPairFile", "avgCoordX", "avgCoordY", "avgCoordZ",
	]
	with (visualization_root / pattern_summary_file).open(mode="w") as summary_handle:
		summary_handle.write(",".join(headers) + "\n")
		for row in pattern_summary_rows:
			summary_handle.write(",".join([
				json.dumps(row[header]) if isinstance(row[header], str) else str(row[header])
				for header in headers
			]) + "\n")

	return representative_manifest, aggregate_manifest, pattern_summary_rows

def exportOccurrenceGraphViews(
		residue_occurrence_file="pattern_occurrences_residue.json",
		graphs_dir="data/graphs",
		visualization_dir="visualization",
		index_file="occurrence_graph_index.json",
		index_html_file="occurrence_graph_index.html",
		path=""):

	with (Path(path)/residue_occurrence_file).open() as residue_occurrences_handle:
		residue_occurrences = json.load(residue_occurrences_handle)

	graphs_root = Path(path) / graphs_dir
	visualization_root = Path(path) / visualization_dir
	html_dir = visualization_root / "graph_occurrences_html"
	html_dir.mkdir(parents=True, exist_ok=True)

	graph_lookup = {}
	for support_dir in sorted(graphs_root.glob("json_*")):
		support = support_dir.name.replace("json_", "")
		for graph_file in support_dir.glob("*.graph.json"):
			graph_id = graph_file.name.split(".")[-3]
			graph_lookup[(support, graph_id)] = graph_file

	index = []
	for occurrence in residue_occurrences:
		graph_key = (str(occurrence["support"]), str(occurrence["graphId"]))
		graph_file = graph_lookup.get(graph_key)
		if graph_file is None:
			continue

		graph_data = json.loads(graph_file.read_text())
		graph_size = {
			"nodeCount": len(graph_data["nodes"]),
			"edgeCount": len(graph_data["links"]),
		}
		html_file = html_dir / (
			f"occurrence_{occurrence['occurrenceId']}_cluster_{occurrence['cluster']}_"
			f"pattern_{occurrence['patternId']}_support_{_safe_name(occurrence['support'])}.html"
		)
		page_data = {
			"occurrence": occurrence,
			"graph": graph_data,
			"graphFile": str(graph_file),
			"graphSize": graph_size,
		}

		html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Occurrence Graph Viewer</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 0;
      background: #f5f1e8;
      color: #1d1a16;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      font-weight: 600;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #d9cfbb;
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 6px 20px rgba(72, 52, 24, 0.06);
    }}
    .graph-panel {{
      background: #fffdf8;
      border: 1px solid #d9cfbb;
      border-radius: 18px;
      padding: 10px;
      margin-bottom: 20px;
    }}
    svg {{
      width: 100%;
      height: 760px;
      display: block;
      background: radial-gradient(circle at top, #fffefb 0%, #f4ecdd 100%);
      border-radius: 14px;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin: 10px 0 0;
      font-size: 14px;
    }}
    .dot {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fffdf8;
      border-radius: 14px;
      overflow: hidden;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #eadfca;
    }}
    th {{
      background: #efe5d2;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Occurrence Graph Viewer</h1>
    <div class="meta" id="meta"></div>
    <div class="graph-panel">
      <svg id="graph" viewBox="0 0 1000 760" preserveAspectRatio="xMidYMid meet"></svg>
		      <div class="legend">
		        <div><span class="dot" style="background:#7CB4BE; border-radius:2px; border:2px solid #26454c;"></span>Protein/source-side node</div>
		        <div><span class="dot" style="background:#9BCE91; border:2px solid #365b2f;"></span>Ligand/target-side node</div>
		        <div><span class="dot" style="background:#e4572e; border-radius:2px; border:2px solid #5d1f10;"></span>Matched protein/source node</div>
		        <div><span class="dot" style="background:#d1498b; border:2px solid #6f1d46;"></span>Matched ligand/target node</div>
		      </div>
    </div>
    <h2>Matched Residues</h2>
    <table>
      <thead>
        <tr>
          <th>Chain</th>
          <th>Residue</th>
          <th>Number</th>
          <th>Matched Atoms</th>
          <th>Role</th>
        </tr>
      </thead>
      <tbody id="residue-table"></tbody>
    </table>
  </div>
  <script>
    const pageData = {json.dumps(page_data)};
    const occurrence = pageData.occurrence;
    const graph = pageData.graph;
    const matchedNodes = new Set(occurrence.matchedGraphNodeIds.map(String));
    const matchedEdges = new Set(
      occurrence.matchedGraphEdgeIds.map(edge => edge.map(String).sort().join('-'))
    );

    const meta = document.getElementById('meta');
    const cards = [
      ['Occurrence', occurrence.occurrenceId],
      ['Cluster / Pattern', `${{occurrence.cluster}} / ${{occurrence.patternId}}`],
      ['Support', occurrence.support],
      ['Graph ID', occurrence.graphId],
      ['Graph Size', `${{pageData.graphSize.nodeCount}} nodes / ${{pageData.graphSize.edgeCount}} edges`],
      ['PDB', occurrence.structureContext.pdbid],
      ['Chain Pair', `${{occurrence.structureContext.chainPair.source}}-${{occurrence.structureContext.chainPair.target}}`],
      ['Graph File', pageData.graphFile]
    ];
    for (const [label, value] of cards) {{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `<strong>${{label}}</strong><br><code>${{value}}</code>`;
      meta.appendChild(card);
    }}

    const residueTable = document.getElementById('residue-table');
    for (const residue of occurrence.matchedResidues) {{
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${{residue.chain}}</td>
        <td>${{residue.residueName}}</td>
        <td>${{residue.residueNumber}}</td>
        <td>${{residue.matchedAtomNames.join(', ')}}</td>
        <td>${{residue.isLigand ? 'target' : 'source'}}</td>
      `;
      residueTable.appendChild(row);
    }}

    const svg = document.getElementById('graph');
    const width = 1000;
    const height = 760;
	    const nodes = [...graph.nodes].sort((a, b) => a.index - b.index);
	    const positions = new Map();
	    const sourceNodes = nodes.filter(node => !node.isLigand);
	    const targetNodes = nodes.filter(node => node.isLigand);
	    const laneY = (index, count) => {{
	      if (count === 1) return height / 2;
	      const top = 110;
	      const bottom = height - 110;
	      return top + ((bottom - top) * index / (count - 1));
	    }};
	    sourceNodes.forEach((node, i) => {{
	      positions.set(String(node.index), {{
	        x: width * 0.28,
	        y: laneY(i, sourceNodes.length)
	      }});
	    }});
	    targetNodes.forEach((node, i) => {{
	      positions.set(String(node.index), {{
	        x: width * 0.72,
	        y: laneY(i, targetNodes.length)
	      }});
	    }});

	    const roleLeft = document.createElementNS('http://www.w3.org/2000/svg', 'text');
	    roleLeft.setAttribute('x', width * 0.28);
	    roleLeft.setAttribute('y', 44);
	    roleLeft.setAttribute('text-anchor', 'middle');
	    roleLeft.setAttribute('font-size', '18');
	    roleLeft.setAttribute('font-weight', '700');
	    roleLeft.setAttribute('fill', '#355d66');
	    roleLeft.textContent = 'Protein / Source';
	    svg.appendChild(roleLeft);

	    const roleRight = document.createElementNS('http://www.w3.org/2000/svg', 'text');
	    roleRight.setAttribute('x', width * 0.72);
	    roleRight.setAttribute('y', 44);
	    roleRight.setAttribute('text-anchor', 'middle');
	    roleRight.setAttribute('font-size', '18');
	    roleRight.setAttribute('font-weight', '700');
	    roleRight.setAttribute('fill', '#4f7a45');
	    roleRight.textContent = 'Ligand / Target';
	    svg.appendChild(roleRight);

    for (const link of graph.links) {{
      const source = positions.get(String(link.source));
      const target = positions.get(String(link.target));
      const edgeKey = [String(link.source), String(link.target)].sort().join('-');
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', source.x);
      line.setAttribute('y1', source.y);
      line.setAttribute('x2', target.x);
      line.setAttribute('y2', target.y);
      line.setAttribute('stroke', matchedEdges.has(edgeKey) ? '#e4572e' : '#baa98a');
      line.setAttribute('stroke-width', matchedEdges.has(edgeKey) ? '4' : '2');
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('opacity', matchedEdges.has(edgeKey) ? '0.95' : '0.7');
      svg.appendChild(line);
    }}

    for (const node of nodes) {{
      const pos = positions.get(String(node.index));
      const isMatched = matchedNodes.has(String(node.index));

	      const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
	      const matchedFill = node.isLigand ? '#d1498b' : '#e4572e';
	      const baseFill = isMatched ? matchedFill : node.color;
	      const glow = document.createElementNS('http://www.w3.org/2000/svg', node.isLigand ? 'circle' : 'rect');
	      if (node.isLigand) {{
	        glow.setAttribute('cx', pos.x);
	        glow.setAttribute('cy', pos.y);
	        glow.setAttribute('r', isMatched ? '24' : '20');
	      }} else {{
	        const glowSize = isMatched ? 46 : 40;
	        glow.setAttribute('x', pos.x - glowSize / 2);
	        glow.setAttribute('y', pos.y - glowSize / 2);
	        glow.setAttribute('width', glowSize);
	        glow.setAttribute('height', glowSize);
	        glow.setAttribute('rx', '6');
	      }}
	      glow.setAttribute('fill', node.isLigand ? 'rgba(155, 206, 145, 0.18)' : 'rgba(124, 180, 190, 0.2)');
	      glow.setAttribute('stroke', 'none');
	      group.appendChild(glow);

	      const nodeShape = document.createElementNS('http://www.w3.org/2000/svg', node.isLigand ? 'circle' : 'rect');
	      if (node.isLigand) {{
	        nodeShape.setAttribute('cx', pos.x);
	        nodeShape.setAttribute('cy', pos.y);
	        nodeShape.setAttribute('r', isMatched ? '19' : '15');
	      }} else {{
	        const size = isMatched ? 36 : 30;
	        nodeShape.setAttribute('x', pos.x - size / 2);
	        nodeShape.setAttribute('y', pos.y - size / 2);
	        nodeShape.setAttribute('width', size);
	        nodeShape.setAttribute('height', size);
	        nodeShape.setAttribute('rx', '5');
	      }}
	      nodeShape.setAttribute('fill', baseFill);
	      nodeShape.setAttribute('stroke', node.isLigand ? '#365b2f' : '#26454c');
	      nodeShape.setAttribute('stroke-width', isMatched ? '3.5' : '2.4');
	      group.appendChild(nodeShape);

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', pos.x);
      label.setAttribute('y', pos.y + 4);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('font-size', '11');
      label.setAttribute('font-weight', '700');
      label.setAttribute('fill', '#fff');
      label.textContent = node.index;
      group.appendChild(label);

      const caption = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      caption.setAttribute('x', pos.x);
      caption.setAttribute('y', pos.y + 34);
      caption.setAttribute('text-anchor', 'middle');
      caption.setAttribute('font-size', '12');
      caption.setAttribute('fill', '#2d2419');
	      caption.textContent = `${{node.isLigand ? 'L' : 'P'}} | ${{node.chain}}:${{node.residueName}}${{node.residueNumber}}:${{node.atomName}}`;
      group.appendChild(caption);

      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = [
        `node=${{node.index}}`,
        `chain=${{node.chain}}`,
        `residue=${{node.residueName}}${{node.residueNumber}}`,
        `atom=${{node.atomName}}`,
        `atomType=${{node.atomType}}`,
	        `role=${{node.isLigand ? 'ligand/target' : 'protein/source'}}`,
	        `matched=${{isMatched}}`
      ].join(' | ');
      group.appendChild(title);

      svg.appendChild(group);
    }}
  </script>
</body>
</html>
"""
		html_file.write_text(html)
		index.append({
			"occurrenceId": occurrence["occurrenceId"],
			"cluster": occurrence["cluster"],
			"support": occurrence["support"],
			"patternId": occurrence["patternId"],
			"graphId": occurrence["graphId"],
			"pdbid": occurrence["structureContext"]["pdbid"],
			"html": html_file.name,
			"graphFile": str(graph_file),
			"graphNodeCount": graph_size["nodeCount"],
			"graphEdgeCount": graph_size["edgeCount"],
			"matchedNodeIds": occurrence["matchedGraphNodeIds"],
			"matchedEdgeIds": occurrence["matchedGraphEdgeIds"],
		})

	with (visualization_root / index_file).open(mode="w") as index_handle:
		json.dump(index, index_handle, indent=4)

	index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Occurrence Graph Index</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 0;
      background: linear-gradient(180deg, #f7f2e8 0%, #efe5d6 100%);
      color: #211b15;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-weight: 600;
    }}
    p {{
      margin: 0 0 18px;
      max-width: 760px;
      line-height: 1.5;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    input {{
      border: 1px solid #cdbda3;
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fffdf8;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 253, 248, 0.94);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 8px 28px rgba(76, 55, 24, 0.08);
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #eadfca;
      font-size: 14px;
      vertical-align: top;
    }}
    th {{
      background: #e9dcc4;
      position: sticky;
      top: 0;
    }}
    a {{
      color: #8d3c1f;
      text-decoration: none;
      font-weight: 600;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Original Graphs With Hotspot Occurrences</h1>
    <p>
      Each row links to the original node-edge graph where a hotspot occurrence was found.
      Open the HTML link to inspect the graph directly in the browser with matched nodes and edges highlighted.
    </p>
    <div class="toolbar">
      <input id="clusterFilter" placeholder="Filter by cluster">
      <input id="supportFilter" placeholder="Filter by support">
      <input id="patternFilter" placeholder="Filter by pattern">
      <input id="pdbFilter" placeholder="Filter by PDB">
      <input id="minNodesFilter" placeholder="Min nodes">
      <input id="maxNodesFilter" placeholder="Max nodes">
      <input id="minEdgesFilter" placeholder="Min edges">
      <input id="maxEdgesFilter" placeholder="Max edges">
    </div>
    <table>
      <thead>
        <tr>
          <th>Occurrence</th>
          <th>Cluster</th>
          <th>Support</th>
          <th>Pattern</th>
          <th>Graph</th>
          <th>Size</th>
          <th>PDB</th>
          <th>HTML View</th>
          <th>Graph JSON</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <script>
    const indexData = {json.dumps(index)};
    const rowsEl = document.getElementById('rows');
    const filters = {{
      cluster: document.getElementById('clusterFilter'),
      support: document.getElementById('supportFilter'),
      pattern: document.getElementById('patternFilter'),
      pdb: document.getElementById('pdbFilter'),
      minNodes: document.getElementById('minNodesFilter'),
      maxNodes: document.getElementById('maxNodesFilter'),
      minEdges: document.getElementById('minEdgesFilter'),
      maxEdges: document.getElementById('maxEdgesFilter'),
    }};

    function render() {{
      rowsEl.innerHTML = '';
      const clusterValue = filters.cluster.value.trim().toLowerCase();
      const supportValue = filters.support.value.trim().toLowerCase();
      const patternValue = filters.pattern.value.trim().toLowerCase();
      const pdbValue = filters.pdb.value.trim().toLowerCase();
      const minNodesValue = Number(filters.minNodes.value);
      const maxNodesValue = Number(filters.maxNodes.value);
      const minEdgesValue = Number(filters.minEdges.value);
      const maxEdgesValue = Number(filters.maxEdges.value);

      for (const item of indexData) {{
        if (clusterValue && String(item.cluster).toLowerCase() !== clusterValue) continue;
        if (supportValue && String(item.support).toLowerCase() !== supportValue) continue;
        if (patternValue && String(item.patternId).toLowerCase() !== patternValue) continue;
        if (pdbValue && String(item.pdbid).toLowerCase() !== pdbValue) continue;
        if (filters.minNodes.value && item.graphNodeCount < minNodesValue) continue;
        if (filters.maxNodes.value && item.graphNodeCount > maxNodesValue) continue;
        if (filters.minEdges.value && item.graphEdgeCount < minEdgesValue) continue;
        if (filters.maxEdges.value && item.graphEdgeCount > maxEdgesValue) continue;

        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${{item.occurrenceId}}</td>
          <td>${{item.cluster}}</td>
          <td>${{item.support}}</td>
          <td>${{item.patternId}}</td>
          <td>${{item.graphId}}</td>
          <td>${{item.graphNodeCount}}N / ${{item.graphEdgeCount}}E</td>
          <td>${{item.pdbid}}</td>
          <td><a href="graph_occurrences_html/${{item.html}}" target="_blank">open graph</a></td>
          <td><code>${{item.graphFile}}</code></td>
        `;
        rowsEl.appendChild(row);
      }}
    }}

    Object.values(filters).forEach(input => input.addEventListener('input', render));
    render();
  </script>
</body>
</html>
"""
	with (visualization_root / index_html_file).open(mode="w") as html_index_handle:
		html_index_handle.write(index_html)

	return index

def _count_maximal_patterns(maximal_file):
	with maximal_file.open() as maximal_handle:
		maximal_data = json.load(maximal_handle)
	return sum(
		len(patterns_at_support)
		for cluster in maximal_data
		for patterns_at_support in cluster.values()
	)

def _collect_results_metrics(results_dir):
	results_dir = Path(results_dir)
	metrics = {
		"resultsDir": str(results_dir),
		"graphsCount": 0,
		"clusterCount": 0,
		"hasGSpanFp": False,
		"maximalPatternCount": 0,
		"legacyOutputs": {},
		"newOutputs": {},
	}

	graphs_file = results_dir / "graphs.txt"
	if graphs_file.exists():
		with graphs_file.open() as graphs_handle:
			metrics["graphsCount"] = len(graphs_handle.read().split("#")) - 1

	clusters_file = results_dir / "clusters.csv"
	if clusters_file.exists():
		clusters = np.loadtxt(clusters_file, dtype=int, delimiter=",")
		clusters = np.atleast_1d(clusters)
		metrics["clusterCount"] = len(set(int(value) for value in clusters.tolist()))

	metrics["hasGSpanFp"] = (results_dir / "gSpan.fp").exists()

	maximal_file = results_dir / "maximal.json"
	if maximal_file.exists():
		metrics["maximalPatternCount"] = _count_maximal_patterns(maximal_file)

	legacy_outputs = [
		"graphs.txt",
		"count_matrix.csv",
		"clusters.csv",
		"gSpan.fp",
		"maximal.json",
		"data",
	]
	new_outputs = [
		"pattern_occurrences.json",
		"pattern_occurrences_residue.json",
		"pattern_hotspots_summary.csv",
		"visualization",
	]
	for name in legacy_outputs:
		metrics["legacyOutputs"][name] = (results_dir / name).exists()
	for name in new_outputs:
		metrics["newOutputs"][name] = (results_dir / name).exists()

	return metrics

def validateRegression(
		current_results_dir,
		baseline_results_dir=None,
		report_file="regression_validation.json",
		summary_file="regression_validation.md"):

	current_results_dir = Path(current_results_dir)
	current_metrics = _collect_results_metrics(current_results_dir)
	report = {
		"current": current_metrics,
		"baseline": None,
		"comparisons": {},
		"checks": {},
	}

	baseline_path = Path(baseline_results_dir) if baseline_results_dir else None
	if baseline_path and baseline_path.exists():
		baseline_metrics = _collect_results_metrics(baseline_path)
		report["baseline"] = baseline_metrics
		report["comparisons"] = {
			"graphsCountUnchanged": current_metrics["graphsCount"] == baseline_metrics["graphsCount"],
			"clusterCountUnchanged": current_metrics["clusterCount"] == baseline_metrics["clusterCount"],
			"hasGSpanFpUnchanged": current_metrics["hasGSpanFp"] == baseline_metrics["hasGSpanFp"],
			"maximalPatternCountUnchanged": (
				current_metrics["maximalPatternCount"] == baseline_metrics["maximalPatternCount"]
			),
		}

	report["checks"] = {
		"legacyOutputsPreserved": all(current_metrics["legacyOutputs"].values()),
		"newOutputsGenerated": all(current_metrics["newOutputs"].values()),
	}
	if report["baseline"] is not None:
		report["checks"]["miningSemanticsStable"] = all(report["comparisons"].values())
	else:
		report["checks"]["miningSemanticsStable"] = None

	with (current_results_dir / report_file).open(mode="w") as report_handle:
		json.dump(report, report_handle, indent=4)

	summary_lines = [
		"# Regression Validation",
		"",
		f"- Current results: `{current_results_dir}`",
		f"- Baseline results: `{baseline_path}`" if baseline_path and baseline_path.exists() else "- Baseline results: not provided",
		"",
		"## Current Metrics",
		f"- Graphs: {current_metrics['graphsCount']}",
		f"- Clusters: {current_metrics['clusterCount']}",
		f"- `gSpan.fp` present: {current_metrics['hasGSpanFp']}",
		f"- Maximal patterns: {current_metrics['maximalPatternCount']}",
		"",
		"## Output Checks",
		f"- Legacy outputs preserved: {report['checks']['legacyOutputsPreserved']}",
		f"- New outputs generated: {report['checks']['newOutputsGenerated']}",
	]
	if report["baseline"] is not None:
		summary_lines.extend([
			"",
			"## Baseline Comparison",
			f"- Graph count unchanged: {report['comparisons']['graphsCountUnchanged']}",
			f"- Cluster count unchanged: {report['comparisons']['clusterCountUnchanged']}",
			f"- `gSpan.fp` presence unchanged: {report['comparisons']['hasGSpanFpUnchanged']}",
			f"- Maximal pattern count unchanged: {report['comparisons']['maximalPatternCountUnchanged']}",
			f"- Mining semantics stable: {report['checks']['miningSemanticsStable']}",
		])

	with (current_results_dir / summary_file).open(mode="w") as summary_handle:
		summary_handle.write("\n".join(summary_lines) + "\n")

	return report

def printMaximalResults(results,file,info=None):
	with open(file,'w') as out:
		sys.stdout = out

		if info:
			for k,v in sorted(info.items()):
				print(k,v)
		n_cluster = 0
		print(type(results))
		for cluster in results:
			print(400*'#')
			print('Cluster:', n_cluster)
			support = 0.1
			for graphs in cluster:
				print(100*'-')
				print("Support: %.1f"%support, end="  ")
				print("N_graphs: ", len(graphs))
				print('{%.1f'%support)
				for g in graphs:
					print(g["graph"].graph)
					print("P_id",g["graph"].graph["id"])
					print(['N%dE%d' %(g["graph"].number_of_nodes(),
						g["graph"].number_of_edges())])
					print(g["graph"].nodes(data=True))
					print(g["graph"].edges(data=True))
				print('}')
				support += 0.1
   
			n_cluster+=1

	


	sys.stdout = sys.__stdout__

def jsonParse(clusters,patterns,atom_types,supports,type_code,typenames,path=""):

	data_dir_name = 'data'
	path = Path(path) / data_dir_name
	
	path.mkdir(parents=True,exist_ok=True)
	
	### Group Info
	graphs = []
	for k,v in sorted(clusters.items()):
		for graph in v:
			graphs.append((k,graph))

	graphs.sort(key=lambda x: int(x[1].graph['id']))

	with (path/'group-info.csv').open(mode="w") as gFile:
		gFile.write('"graph","group","pdb","chain","ligand"\n')
		for g in graphs:
			k,g = g
			g = g.graph
			gFile.write('%d,%d,"%s","%s","%s"\n'%(int(g['id']),k+1,g['pdbid'],g['source'],g['target']))


	### Graphs Files
	with open(typenames,'r') as typenames_file:
		typenames = json.load(typenames_file)

	
	typeInt = {
		'ACP' : 2,
		'ARM' : 3,
		'DON' : 5,
		'HPB' : 7,
		'POS' : 11,
		'NEG' : 13,
		'SSB' :	17,
		"HYDROPHOBIC": 29,
		"SALT_BRIDGE": 23,
		"ARM_STACK": 19,
		"H_BOND": 31,
		"REPULSIVE": 37,
		"SS_BRIDGE": 41
	}

	bin_to_prime = {
		1 : 2,
		2 : 3,
		4 : 5,
		8 : 7,
		32 : 11,
		16 : 13,
		256 : 23,
		2048 : 19,
		128 : 17,
		512 : 29,
		1024 : 31
	}

	# colors = list(reversed(["a6cee3","1f78b4","b2df8a","33a02c",'fb9a99','e31a1c',
	# 'fdbf6f','ff7f00','cab2d6','6a3d9a','ffff99','b15928']))
	chain_color = "#7CB4BE"
	lig_color = "#9BCE91"

	node_types = dict()
	int_types = dict()
	colormap = dict()
	typeMap = dict()
	atomTypeInt = 0
	intTypeInt = 0
	count = dict()

	graph_list_file = (path/"list-graphs").open(mode='w')

	########################## Generate graph files ##########################
	graphs_path = Path("graphs")
	(path / graphs_path).mkdir(parents=True,exist_ok=True)

	for s in supports:

		dir_name = 'json_%s'%s
		dir_path = graphs_path / dir_name
		
		(path/dir_path).mkdir(parents=True,exist_ok=True)

		with cd(path/dir_path):

			j_graphs = []
			for k,v in sorted(clusters.items()):
				for graphs in v:
					keys = ['pdbid','source','id']
					g = {'data':{'nodes': [], 'links': []},
						'meta':tuple([k+1]+[graphs.graph[x] for x in keys])}
	          		
					### Nodes
					for node in graphs.nodes():
						temp_node = dict()
						temp_node['index'] = int(node)

						node = graphs.nodes[node]

						temp_node['patterns'] = list(node['patterns'][s])
						temp_node['chain'] = node['chain']
						temp_node['atomName'] = node['atomName']
						temp_node['residueNumber'] = str(node['residueNumber'])
						temp_node['residueName'] = node['residueName']
						temp_node['isLigand'] = True if node['isLigand'] else False
						temp_node['atomType'] = atom_types[node['residueName']][node['atomName']]
						temp_node['color'] = lig_color if temp_node['isLigand'] else chain_color

						atomTypeInt = reduce(lambda x,y: x*y,[i for i in map(lambda x: typeInt[x],temp_node['atomType'])])
						atomType = temp_node['atomType'] = '/'.join(sorted([i for i in map(lambda x: typenames[x],temp_node['atomType'])]))
						temp_node['atomTypeInt'] = str(atomTypeInt)

						g['data']['nodes'].append(temp_node)

					g['data']['nodes'].sort(key=lambda x: x['index'])

					### Edges (links)
					nodes = graphs.nodes
					for edge in graphs.edges(data=True):
						
						temp_edge = dict()
						temp_edge['patterns'] = list(edge[2]['patterns'][s])

						if nodes[edge[0]]["isLigand"]:
							temp_edge['source'] = int(edge[1])
							temp_edge['target'] = int(edge[0])

						else:
							temp_edge['source'] = int(edge[0])
							temp_edge['target'] = int(edge[1])

						interactionType = type_code.type(edge[2]['type'])

						temp_edge['interactionType'] = "/".join(
							typenames[t] for t in interactionType)
						temp_edge['distance'] = str(edge[2]['distance'])
						temp_edge['interactionTypeInt'] = str(reduce(lambda x,y: x*y,
								[typeInt[t] for t in interactionType]))
						g['data']['links'].append(temp_edge)
					j_graphs.append(g)

			### Write json graph files			
			for g in j_graphs:
				fname = "g%d.%s.%s.%d.graph.json"%(g['meta'])
				f_path = dir_path/fname

				graph_list_file.write(str(f_path) + '\n')
				with open(fname,'w') as out_json:
				 	out_json.write(json.dumps(g['data'],indent=4))

	graph_list_file.close()

	########################## Generate mapping files ##########################
	mappings_array = []
	
	for key,cl in sorted(clusters.items()):
		
		for sup,graphs in patterns[key].items():
	
			patterns_array = []
			for g in sorted(graphs,key=lambda x: int(x.graph["id"])):
				patterns_array.append({
					"entranceGraphs": g.graph["ocur"],
					"patternLabel" : str(g.graph['id']),
					"patternSize" : len(g)
					})
	
			mappings_array.append({
				"support" : sup,
				"patterns" : patterns_array,
				"group" : str(key+1)
				})


	with (path/"files_mapping.json").open(mode="w") as files_mapping_file:
		files_mapping_file.write(json.dumps(mappings_array,indent=4))

	########################## Generate pattern files ##########################
	patterns_list_file = (path/"list-patterns-graphs").open("w")

	dir_name = 'patterns'
	dir_path = path / dir_name
	dir_path.mkdir(parents=True,exist_ok=True)
	
	with cd(dir_path):
		for s in supports:
			for k in clusters:
				pattern_group = patterns[k]
				for graph in pattern_group[s]:
					p = {'nodes': [], 'links': [],'graphproperties':{}}

					for node in graph.nodes():
						temp_node = {}	
						temp_node['index'] = int(node)

						atomType = type_code.type(graph.nodes[node]['type'])
						atomTypeInt = reduce(lambda x,y: x*y,[typeInt[i] for i in atomType])
						atomType = '/'.join(sorted(map(lambda x: typenames[x], atomType)))
						temp_node['atomType'] = atomType

						temp_node['atomTypeInt'] = str(atomTypeInt)
						p['nodes'].append(temp_node)

					p['nodes'].sort(key = lambda x: x["index"])

					for edge in graph.edges(data=True):
						temp_edge = {}
						temp_edge['source'] = int(edge[0])
						temp_edge['target'] = int(edge[1])
						interactionType = type_code.type(edge[2]['type'])
						temp_edge['interactionType'] = "/".join(
							typenames[t] for t in interactionType)
						temp_edge['interactionTypeInt'] = str(reduce(lambda x,y: x*y,
								[typeInt[t] for t in interactionType]))
						p['links'].append(temp_edge)
						
					p['graphproperties']['inputgraphs'] = [clusters[k][int(i)].graph["id"] for i in graph.graph['ocur']]
					fname = "g%d.gsp_%s.maximal.fp.patternIndex%d.json" % (k+1,s,graph.graph["id"])

					patterns_list_file.write('patterns/'+fname+"\n")

					with open(fname,'w') as out_json:
						out_json.write(json.dumps(p,indent=4))

	patterns_list_file.close()

	########################## Vertex number file ##########################
	vert_number_file = str(path/'vert_number.csv')

	vert_number_array = [["group","support","patternSize","occurrences"]]
	for key,pattern_group in enumerate(patterns):
		for sup,graphs in patterns[key].items():
			patterns_info = {}
			for g in graphs:
				patterns_info[len(g)] = patterns_info.get(len(g),0) + 1
			#print(patterns_info)
			for p_key,p_val in sorted(patterns_info.items()):
				vert_number_array.append([key+1,sup,p_key,p_val])

	np.savetxt(vert_number_file, vert_number_array, delimiter=',',fmt="%s")

def maximalCount(patterns,node_labels,edge_labels,typenames,type_code,path=""):
	data_dir_name = 'data'
	path = Path(path) / data_dir_name
	filename = 'count_atoms_and_interactions.csv'
	fpath = str(path/filename)

	
	p_edge_labels = set()
	p_node_labels = set()
	for g_patterns in patterns:
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
	print(edge_single_labels)
	
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
	
	labels = { k:v for v,k in enumerate(node_labels + edge_single_labels,2)}
	columns = len(labels)

	r_edge_single_labels = [ i for i in reversed(edge_single_labels)]

	count_matrix = ["/".join(typenames[t] for t in type_code.type(l)) for l in labels]
	for i in range(len(count_matrix)):
		if i < len(node_labels):
			count_matrix[i] = "atoms" + count_matrix[i]
		else:
			count_matrix[i] = "inter" + count_matrix[i]

	count_matrix = [['group','support'] + count_matrix] 
	
	for cl_idx in range(len(patterns)):
		for min_sup,v in patterns[cl_idx].items():
			line = [cl_idx,min_sup] + [0]*columns
	
			for graph in v:
				types = nx.get_node_attributes(graph,"type").values()
				for t in types:
					line[labels[t]] += 1

				for attr in nx.get_edge_attributes(graph,"type").values():
					for t in r_edge_single_labels:
						if t <= attr:
							attr -= t
							line[labels[t]] += 1
			count_matrix.append(line)
	count_matrix = np.array(count_matrix)
	
	with Path(fpath).open(mode="w") as file:
		np.savetxt(file, count_matrix, delimiter=',',fmt="%s")
