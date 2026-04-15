load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wa1.ent", occ_108_c0_p0_s0.8
hide everything, occ_108_c0_p0_s0.8
show cartoon, occ_108_c0_p0_s0.8 and chain A+B
color palegreen, occ_108_c0_p0_s0.8 and chain A
color lightblue, occ_108_c0_p0_s0.8 and chain B
select hotspot_source, occ_108_c0_p0_s0.8 and ((chain A and resi 24))
select hotspot_target, occ_108_c0_p0_s0.8 and ((chain B and resi 475))
select hotspot_all, occ_108_c0_p0_s0.8 and ((chain A and resi 24) or (chain B and resi 475))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_108_c0_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_108
set_name hotspot_source, hotspot_source_108
set_name hotspot_target, hotspot_target_108
bg_color white
# patternId=0 support=0.8 graphId=259
