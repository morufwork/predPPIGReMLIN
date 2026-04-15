load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7pki.ent", occ_2296_c4_p0_s0.9
hide everything, occ_2296_c4_p0_s0.9
show cartoon, occ_2296_c4_p0_s0.9 and chain A+E
color palegreen, occ_2296_c4_p0_s0.9 and chain A
color lightblue, occ_2296_c4_p0_s0.9 and chain E
select hotspot_source, occ_2296_c4_p0_s0.9 and ((chain A and resi 27))
select hotspot_target, occ_2296_c4_p0_s0.9 and ((chain E and resi 485))
select hotspot_all, occ_2296_c4_p0_s0.9 and ((chain A and resi 27) or (chain E and resi 485))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2296_c4_p0_s0.9 and chain A+E
set_name hotspot_all, hotspot_occurrence_2296
set_name hotspot_source, hotspot_source_2296
set_name hotspot_target, hotspot_target_2296
bg_color white
# patternId=0 support=0.9 graphId=168
