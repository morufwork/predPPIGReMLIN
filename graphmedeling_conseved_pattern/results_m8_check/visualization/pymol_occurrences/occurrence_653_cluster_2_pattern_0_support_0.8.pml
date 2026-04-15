load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_653_c2_p0_s0.8
hide everything, occ_653_c2_p0_s0.8
show cartoon, occ_653_c2_p0_s0.8 and chain E+A
color palegreen, occ_653_c2_p0_s0.8 and chain E
color lightblue, occ_653_c2_p0_s0.8 and chain A
select hotspot_source, occ_653_c2_p0_s0.8 and ((chain E and resi 484))
select hotspot_target, occ_653_c2_p0_s0.8 and ((chain A and resi 31))
select hotspot_all, occ_653_c2_p0_s0.8 and ((chain A and resi 31) or (chain E and resi 484))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_653_c2_p0_s0.8 and chain E+A
set_name hotspot_all, hotspot_occurrence_653
set_name hotspot_source, hotspot_source_653
set_name hotspot_target, hotspot_target_653
bg_color white
# patternId=0 support=0.8 graphId=149
