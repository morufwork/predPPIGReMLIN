load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7ekc.ent", occ_217_c0_p0_s1.0
hide everything, occ_217_c0_p0_s1.0
show cartoon, occ_217_c0_p0_s1.0 and chain A+B
color palegreen, occ_217_c0_p0_s1.0 and chain A
color lightblue, occ_217_c0_p0_s1.0 and chain B
select hotspot_source, occ_217_c0_p0_s1.0 and ((chain A and resi 27))
select hotspot_target, occ_217_c0_p0_s1.0 and ((chain B and resi 456))
select hotspot_all, occ_217_c0_p0_s1.0 and ((chain A and resi 27) or (chain B and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_217_c0_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_217
set_name hotspot_source, hotspot_source_217
set_name hotspot_target, hotspot_target_217
bg_color white
# patternId=0 support=1.0 graphId=87
