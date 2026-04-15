load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_25_c0_p0_s0.7
hide everything, occ_25_c0_p0_s0.7
show cartoon, occ_25_c0_p0_s0.7 and chain E+A
color palegreen, occ_25_c0_p0_s0.7 and chain E
color lightblue, occ_25_c0_p0_s0.7 and chain A
select hotspot_source, occ_25_c0_p0_s0.7 and ((chain E and resi 456))
select hotspot_target, occ_25_c0_p0_s0.7 and ((chain A and resi 31))
select hotspot_all, occ_25_c0_p0_s0.7 and ((chain A and resi 31) or (chain E and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_25_c0_p0_s0.7 and chain E+A
set_name hotspot_all, hotspot_occurrence_25
set_name hotspot_source, hotspot_source_25
set_name hotspot_target, hotspot_target_25
bg_color white
# patternId=0 support=0.7 graphId=148
