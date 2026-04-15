load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpb.ent", occ_1880_c4_p0_s0.7
hide everything, occ_1880_c4_p0_s0.7
show cartoon, occ_1880_c4_p0_s0.7 and chain A+D
color palegreen, occ_1880_c4_p0_s0.7 and chain A
color lightblue, occ_1880_c4_p0_s0.7 and chain D
select hotspot_source, occ_1880_c4_p0_s0.7 and ((chain A and resi 453) or (chain A and resi 470))
select hotspot_target, occ_1880_c4_p0_s0.7 and ((chain D and resi 27))
select hotspot_all, occ_1880_c4_p0_s0.7 and ((chain A and resi 453) or (chain A and resi 470) or (chain D and resi 27))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1880_c4_p0_s0.7 and chain A+D
set_name hotspot_all, hotspot_occurrence_1880
set_name hotspot_source, hotspot_source_1880
set_name hotspot_target, hotspot_target_1880
bg_color white
# patternId=0 support=0.7 graphId=297
