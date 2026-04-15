load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xoc.ent", occ_1346_c3_p0_s0.7
hide everything, occ_1346_c3_p0_s0.7
show cartoon, occ_1346_c3_p0_s0.7 and chain D+A
color palegreen, occ_1346_c3_p0_s0.7 and chain D
color lightblue, occ_1346_c3_p0_s0.7 and chain A
select hotspot_source, occ_1346_c3_p0_s0.7 and ((chain D and resi 42))
select hotspot_target, occ_1346_c3_p0_s0.7 and ((chain A and resi 449))
select hotspot_all, occ_1346_c3_p0_s0.7 and ((chain A and resi 449) or (chain D and resi 42))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1346_c3_p0_s0.7 and chain D+A
set_name hotspot_all, hotspot_occurrence_1346
set_name hotspot_source, hotspot_source_1346
set_name hotspot_target, hotspot_target_1346
bg_color white
# patternId=0 support=0.7 graphId=360
