load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_159_c0_p0_s0.9
hide everything, occ_159_c0_p0_s0.9
show cartoon, occ_159_c0_p0_s0.9 and chain E+A
color palegreen, occ_159_c0_p0_s0.9 and chain E
color lightblue, occ_159_c0_p0_s0.9 and chain A
select hotspot_source, occ_159_c0_p0_s0.9 and ((chain E and resi 456))
select hotspot_target, occ_159_c0_p0_s0.9 and ((chain A and resi 31))
select hotspot_all, occ_159_c0_p0_s0.9 and ((chain A and resi 31) or (chain E and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_159_c0_p0_s0.9 and chain E+A
set_name hotspot_all, hotspot_occurrence_159
set_name hotspot_source, hotspot_source_159
set_name hotspot_target, hotspot_target_159
bg_color white
# patternId=0 support=0.9 graphId=148
