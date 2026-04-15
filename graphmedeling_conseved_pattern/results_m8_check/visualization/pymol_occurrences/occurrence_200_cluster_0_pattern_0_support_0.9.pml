load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dm8.ent", occ_200_c0_p0_s0.9
hide everything, occ_200_c0_p0_s0.9
show cartoon, occ_200_c0_p0_s0.9 and chain A+D
color palegreen, occ_200_c0_p0_s0.9 and chain A
color lightblue, occ_200_c0_p0_s0.9 and chain D
select hotspot_source, occ_200_c0_p0_s0.9 and ((chain A and resi 456))
select hotspot_target, occ_200_c0_p0_s0.9 and ((chain D and resi 31))
select hotspot_all, occ_200_c0_p0_s0.9 and ((chain A and resi 456) or (chain D and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_200_c0_p0_s0.9 and chain A+D
set_name hotspot_all, hotspot_occurrence_200
set_name hotspot_source, hotspot_source_200
set_name hotspot_target, hotspot_target_200
bg_color white
# patternId=0 support=0.9 graphId=382
