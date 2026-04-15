load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpb.ent", occ_1598_c3_p0_s0.9
hide everything, occ_1598_c3_p0_s0.9
show cartoon, occ_1598_c3_p0_s0.9 and chain A+D
color palegreen, occ_1598_c3_p0_s0.9 and chain A
color lightblue, occ_1598_c3_p0_s0.9 and chain D
select hotspot_source, occ_1598_c3_p0_s0.9 and ((chain A and resi 484))
select hotspot_target, occ_1598_c3_p0_s0.9 and ((chain D and resi 24))
select hotspot_all, occ_1598_c3_p0_s0.9 and ((chain A and resi 484) or (chain D and resi 24))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1598_c3_p0_s0.9 and chain A+D
set_name hotspot_all, hotspot_occurrence_1598
set_name hotspot_source, hotspot_source_1598
set_name hotspot_target, hotspot_target_1598
bg_color white
# patternId=0 support=0.9 graphId=301
