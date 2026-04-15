load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sxy.ent", occ_2888_c5_p0_s1.0
hide everything, occ_2888_c5_p0_s1.0
show cartoon, occ_2888_c5_p0_s1.0 and chain B+E
color palegreen, occ_2888_c5_p0_s1.0 and chain B
color lightblue, occ_2888_c5_p0_s1.0 and chain E
select hotspot_source, occ_2888_c5_p0_s1.0 and ((chain B and resi 486))
select hotspot_target, occ_2888_c5_p0_s1.0 and ((chain E and resi 82))
select hotspot_all, occ_2888_c5_p0_s1.0 and ((chain B and resi 486) or (chain E and resi 82))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2888_c5_p0_s1.0 and chain B+E
set_name hotspot_all, hotspot_occurrence_2888
set_name hotspot_source, hotspot_source_2888
set_name hotspot_target, hotspot_target_2888
bg_color white
# patternId=0 support=1.0 graphId=192
