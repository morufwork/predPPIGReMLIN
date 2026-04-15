load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_1411_c3_p0_s0.8
hide everything, occ_1411_c3_p0_s0.8
show cartoon, occ_1411_c3_p0_s0.8 and chain E+A
color palegreen, occ_1411_c3_p0_s0.8 and chain E
color lightblue, occ_1411_c3_p0_s0.8 and chain A
select hotspot_source, occ_1411_c3_p0_s0.8 and ((chain E and resi 487))
select hotspot_target, occ_1411_c3_p0_s0.8 and ((chain A and resi 83))
select hotspot_all, occ_1411_c3_p0_s0.8 and ((chain A and resi 83) or (chain E and resi 487))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1411_c3_p0_s0.8 and chain E+A
set_name hotspot_all, hotspot_occurrence_1411
set_name hotspot_source, hotspot_source_1411
set_name hotspot_target, hotspot_target_1411
bg_color white
# patternId=0 support=0.8 graphId=151
