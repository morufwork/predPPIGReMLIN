load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w9i.ent", occ_899_c2_p0_s0.9
hide everything, occ_899_c2_p0_s0.9
show cartoon, occ_899_c2_p0_s0.9 and chain A+E
color palegreen, occ_899_c2_p0_s0.9 and chain A
color lightblue, occ_899_c2_p0_s0.9 and chain E
select hotspot_source, occ_899_c2_p0_s0.9 and ((chain A and resi 31))
select hotspot_target, occ_899_c2_p0_s0.9 and ((chain E and resi 484))
select hotspot_all, occ_899_c2_p0_s0.9 and ((chain A and resi 31) or (chain E and resi 484))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_899_c2_p0_s0.9 and chain A+E
set_name hotspot_all, hotspot_occurrence_899
set_name hotspot_source, hotspot_source_899
set_name hotspot_target, hotspot_target_899
bg_color white
# patternId=0 support=0.9 graphId=252
