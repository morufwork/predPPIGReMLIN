load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_856_c2_p0_s0.9
hide everything, occ_856_c2_p0_s0.9
show cartoon, occ_856_c2_p0_s0.9 and chain E+A
color palegreen, occ_856_c2_p0_s0.9 and chain E
color lightblue, occ_856_c2_p0_s0.9 and chain A
select hotspot_source, occ_856_c2_p0_s0.9 and ((chain E and resi 417))
select hotspot_target, occ_856_c2_p0_s0.9 and ((chain A and resi 30))
select hotspot_all, occ_856_c2_p0_s0.9 and ((chain A and resi 30) or (chain E and resi 417))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_856_c2_p0_s0.9 and chain E+A
set_name hotspot_all, hotspot_occurrence_856
set_name hotspot_source, hotspot_source_856
set_name hotspot_target, hotspot_target_856
bg_color white
# patternId=0 support=0.9 graphId=145
