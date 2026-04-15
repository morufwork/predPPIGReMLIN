load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_1704_c3_p0_s1.0
hide everything, occ_1704_c3_p0_s1.0
show cartoon, occ_1704_c3_p0_s1.0 and chain B+E
color palegreen, occ_1704_c3_p0_s1.0 and chain B
color lightblue, occ_1704_c3_p0_s1.0 and chain E
select hotspot_source, occ_1704_c3_p0_s1.0 and ((chain B and resi 487))
select hotspot_target, occ_1704_c3_p0_s1.0 and ((chain E and resi 83))
select hotspot_all, occ_1704_c3_p0_s1.0 and ((chain B and resi 487) or (chain E and resi 83))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1704_c3_p0_s1.0 and chain B+E
set_name hotspot_all, hotspot_occurrence_1704
set_name hotspot_source, hotspot_source_1704
set_name hotspot_target, hotspot_target_1704
bg_color white
# patternId=0 support=1.0 graphId=205
