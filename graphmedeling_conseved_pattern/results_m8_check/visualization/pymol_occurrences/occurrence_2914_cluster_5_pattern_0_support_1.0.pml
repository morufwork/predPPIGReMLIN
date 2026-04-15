load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpb.ent", occ_2914_c5_p0_s1.0
hide everything, occ_2914_c5_p0_s1.0
show cartoon, occ_2914_c5_p0_s1.0 and chain A+D
color palegreen, occ_2914_c5_p0_s1.0 and chain A
color lightblue, occ_2914_c5_p0_s1.0 and chain D
select hotspot_source, occ_2914_c5_p0_s1.0 and ((chain A and resi 486))
select hotspot_target, occ_2914_c5_p0_s1.0 and ((chain D and resi 31))
select hotspot_all, occ_2914_c5_p0_s1.0 and ((chain A and resi 486) or (chain D and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2914_c5_p0_s1.0 and chain A+D
set_name hotspot_all, hotspot_occurrence_2914
set_name hotspot_source, hotspot_source_2914
set_name hotspot_target, hotspot_target_2914
bg_color white
# patternId=0 support=1.0 graphId=302
