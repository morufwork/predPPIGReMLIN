load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_167_c0_p0_s0.9
hide everything, occ_167_c0_p0_s0.9
show cartoon, occ_167_c0_p0_s0.9 and chain B+E
color palegreen, occ_167_c0_p0_s0.9 and chain B
color lightblue, occ_167_c0_p0_s0.9 and chain E
select hotspot_source, occ_167_c0_p0_s0.9 and ((chain B and resi 498))
select hotspot_target, occ_167_c0_p0_s0.9 and ((chain E and resi 41))
select hotspot_all, occ_167_c0_p0_s0.9 and ((chain B and resi 498) or (chain E and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_167_c0_p0_s0.9 and chain B+E
set_name hotspot_all, hotspot_occurrence_167
set_name hotspot_source, hotspot_source_167
set_name hotspot_target, hotspot_target_167
bg_color white
# patternId=0 support=0.9 graphId=206
