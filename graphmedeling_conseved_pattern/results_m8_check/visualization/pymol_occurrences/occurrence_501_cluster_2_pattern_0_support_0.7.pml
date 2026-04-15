load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpa.ent", occ_501_c2_p0_s0.7
hide everything, occ_501_c2_p0_s0.7
show cartoon, occ_501_c2_p0_s0.7 and chain A+D
color palegreen, occ_501_c2_p0_s0.7 and chain A
color lightblue, occ_501_c2_p0_s0.7 and chain D
select hotspot_source, occ_501_c2_p0_s0.7 and ((chain A and resi 400))
select hotspot_target, occ_501_c2_p0_s0.7 and ((chain D and resi 37))
select hotspot_all, occ_501_c2_p0_s0.7 and ((chain A and resi 400) or (chain D and resi 37))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_501_c2_p0_s0.7 and chain A+D
set_name hotspot_all, hotspot_occurrence_501
set_name hotspot_source, hotspot_source_501
set_name hotspot_target, hotspot_target_501
bg_color white
# patternId=0 support=0.7 graphId=285
