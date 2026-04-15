load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpa.ent", occ_1114_c2_p0_s1.0
hide everything, occ_1114_c2_p0_s1.0
show cartoon, occ_1114_c2_p0_s1.0 and chain A+D
color palegreen, occ_1114_c2_p0_s1.0 and chain A
color lightblue, occ_1114_c2_p0_s1.0 and chain D
select hotspot_source, occ_1114_c2_p0_s1.0 and ((chain A and resi 495))
select hotspot_target, occ_1114_c2_p0_s1.0 and ((chain D and resi 38))
select hotspot_all, occ_1114_c2_p0_s1.0 and ((chain A and resi 495) or (chain D and resi 38))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1114_c2_p0_s1.0 and chain A+D
set_name hotspot_all, hotspot_occurrence_1114
set_name hotspot_source, hotspot_source_1114
set_name hotspot_target, hotspot_target_1114
bg_color white
# patternId=0 support=1.0 graphId=293
