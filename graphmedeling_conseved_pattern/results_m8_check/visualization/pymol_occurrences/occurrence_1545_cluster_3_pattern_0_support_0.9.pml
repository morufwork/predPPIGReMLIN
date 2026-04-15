load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_1545_c3_p0_s0.9
hide everything, occ_1545_c3_p0_s0.9
show cartoon, occ_1545_c3_p0_s0.9 and chain E+A
color palegreen, occ_1545_c3_p0_s0.9 and chain E
color lightblue, occ_1545_c3_p0_s0.9 and chain A
select hotspot_source, occ_1545_c3_p0_s0.9 and ((chain E and resi 498))
select hotspot_target, occ_1545_c3_p0_s0.9 and ((chain A and resi 38))
select hotspot_all, occ_1545_c3_p0_s0.9 and ((chain A and resi 38) or (chain E and resi 498))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1545_c3_p0_s0.9 and chain E+A
set_name hotspot_all, hotspot_occurrence_1545
set_name hotspot_source, hotspot_source_1545
set_name hotspot_target, hotspot_target_1545
bg_color white
# patternId=0 support=0.9 graphId=146
