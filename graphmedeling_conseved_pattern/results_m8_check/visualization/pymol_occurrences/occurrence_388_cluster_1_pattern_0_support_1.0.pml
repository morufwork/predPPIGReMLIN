load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_388_c1_p0_s1.0
hide everything, occ_388_c1_p0_s1.0
show cartoon, occ_388_c1_p0_s1.0 and chain B+E
color palegreen, occ_388_c1_p0_s1.0 and chain B
color lightblue, occ_388_c1_p0_s1.0 and chain E
select hotspot_source, occ_388_c1_p0_s1.0 and ((chain B and resi 500))
select hotspot_target, occ_388_c1_p0_s1.0 and ((chain E and resi 41))
select hotspot_all, occ_388_c1_p0_s1.0 and ((chain B and resi 500) or (chain E and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_388_c1_p0_s1.0 and chain B+E
set_name hotspot_all, hotspot_occurrence_388
set_name hotspot_source, hotspot_source_388
set_name hotspot_target, hotspot_target_388
bg_color white
# patternId=0 support=1.0 graphId=207
