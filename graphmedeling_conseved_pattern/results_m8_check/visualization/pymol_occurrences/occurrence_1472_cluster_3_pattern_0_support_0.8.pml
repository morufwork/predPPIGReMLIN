load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xo6.ent", occ_1472_c3_p0_s0.8
hide everything, occ_1472_c3_p0_s0.8
show cartoon, occ_1472_c3_p0_s0.8 and chain D+A
color palegreen, occ_1472_c3_p0_s0.8 and chain D
color lightblue, occ_1472_c3_p0_s0.8 and chain A
select hotspot_source, occ_1472_c3_p0_s0.8 and ((chain D and resi 31))
select hotspot_target, occ_1472_c3_p0_s0.8 and ((chain A and resi 493))
select hotspot_all, occ_1472_c3_p0_s0.8 and ((chain A and resi 493) or (chain D and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1472_c3_p0_s0.8 and chain D+A
set_name hotspot_all, hotspot_occurrence_1472
set_name hotspot_source, hotspot_source_1472
set_name hotspot_target, hotspot_target_1472
bg_color white
# patternId=0 support=0.8 graphId=336
