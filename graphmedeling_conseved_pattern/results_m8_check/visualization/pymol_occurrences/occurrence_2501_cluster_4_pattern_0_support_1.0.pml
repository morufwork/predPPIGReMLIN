load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_2501_c4_p0_s1.0
hide everything, occ_2501_c4_p0_s1.0
show cartoon, occ_2501_c4_p0_s1.0 and chain E+A
color palegreen, occ_2501_c4_p0_s1.0 and chain E
color lightblue, occ_2501_c4_p0_s1.0 and chain A
select hotspot_source, occ_2501_c4_p0_s1.0 and ((chain E and resi 456) or (chain E and resi 489))
select hotspot_target, occ_2501_c4_p0_s1.0 and ((chain A and resi 27))
select hotspot_all, occ_2501_c4_p0_s1.0 and ((chain A and resi 27) or (chain E and resi 456) or (chain E and resi 489))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2501_c4_p0_s1.0 and chain E+A
set_name hotspot_all, hotspot_occurrence_2501
set_name hotspot_source, hotspot_source_2501
set_name hotspot_target, hotspot_target_2501
bg_color white
# patternId=0 support=1.0 graphId=147
