load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xo9.ent", occ_1610_c3_p0_s0.9
hide everything, occ_1610_c3_p0_s0.9
show cartoon, occ_1610_c3_p0_s0.9 and chain A+D
color palegreen, occ_1610_c3_p0_s0.9 and chain A
color lightblue, occ_1610_c3_p0_s0.9 and chain D
select hotspot_source, occ_1610_c3_p0_s0.9 and ((chain A and resi 449))
select hotspot_target, occ_1610_c3_p0_s0.9 and ((chain D and resi 42))
select hotspot_all, occ_1610_c3_p0_s0.9 and ((chain A and resi 449) or (chain D and resi 42))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1610_c3_p0_s0.9 and chain A+D
set_name hotspot_all, hotspot_occurrence_1610
set_name hotspot_source, hotspot_source_1610
set_name hotspot_target, hotspot_target_1610
bg_color white
# patternId=0 support=0.9 graphId=342
