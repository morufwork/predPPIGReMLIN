load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xo6.ent", occ_1339_c3_p0_s0.7
hide everything, occ_1339_c3_p0_s0.7
show cartoon, occ_1339_c3_p0_s0.7 and chain D+A
color palegreen, occ_1339_c3_p0_s0.7 and chain D
color lightblue, occ_1339_c3_p0_s0.7 and chain A
select hotspot_source, occ_1339_c3_p0_s0.7 and ((chain D and resi 41))
select hotspot_target, occ_1339_c3_p0_s0.7 and ((chain A and resi 500))
select hotspot_all, occ_1339_c3_p0_s0.7 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1339_c3_p0_s0.7 and chain D+A
set_name hotspot_all, hotspot_occurrence_1339
set_name hotspot_source, hotspot_source_1339
set_name hotspot_target, hotspot_target_1339
bg_color white
# patternId=0 support=0.7 graphId=339
