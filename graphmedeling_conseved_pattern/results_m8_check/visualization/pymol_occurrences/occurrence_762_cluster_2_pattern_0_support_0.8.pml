load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dm6.ent", occ_762_c2_p0_s0.8
hide everything, occ_762_c2_p0_s0.8
show cartoon, occ_762_c2_p0_s0.8 and chain A+D
color palegreen, occ_762_c2_p0_s0.8 and chain A
color lightblue, occ_762_c2_p0_s0.8 and chain D
select hotspot_source, occ_762_c2_p0_s0.8 and ((chain A and resi 403))
select hotspot_target, occ_762_c2_p0_s0.8 and ((chain D and resi 37))
select hotspot_all, occ_762_c2_p0_s0.8 and ((chain A and resi 403) or (chain D and resi 37))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_762_c2_p0_s0.8 and chain A+D
set_name hotspot_all, hotspot_occurrence_762
set_name hotspot_source, hotspot_source_762
set_name hotspot_target, hotspot_target_762
bg_color white
# patternId=0 support=0.8 graphId=371
