load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6m0j.ent", occ_405_c2_p0_s0.7
hide everything, occ_405_c2_p0_s0.7
show cartoon, occ_405_c2_p0_s0.7 and chain A+E
color palegreen, occ_405_c2_p0_s0.7 and chain A
color lightblue, occ_405_c2_p0_s0.7 and chain E
select hotspot_source, occ_405_c2_p0_s0.7 and ((chain A and resi 31))
select hotspot_target, occ_405_c2_p0_s0.7 and ((chain E and resi 484))
select hotspot_all, occ_405_c2_p0_s0.7 and ((chain A and resi 31) or (chain E and resi 484))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_405_c2_p0_s0.7 and chain A+E
set_name hotspot_all, hotspot_occurrence_405
set_name hotspot_source, hotspot_source_405
set_name hotspot_target, hotspot_target_405
bg_color white
# patternId=0 support=0.7 graphId=16
