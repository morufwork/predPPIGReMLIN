load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wse.ent", occ_57_c0_p0_s0.7
hide everything, occ_57_c0_p0_s0.7
show cartoon, occ_57_c0_p0_s0.7 and chain A+B
color palegreen, occ_57_c0_p0_s0.7 and chain A
color lightblue, occ_57_c0_p0_s0.7 and chain B
select hotspot_source, occ_57_c0_p0_s0.7 and ((chain A and resi 352))
select hotspot_target, occ_57_c0_p0_s0.7 and ((chain B and resi 505))
select hotspot_all, occ_57_c0_p0_s0.7 and ((chain A and resi 352) or (chain B and resi 505))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_57_c0_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_57
set_name hotspot_source, hotspot_source_57
set_name hotspot_target, hotspot_target_57
bg_color white
# patternId=0 support=0.7 graphId=322
