load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sxy.ent", occ_463_c2_p0_s0.7
hide everything, occ_463_c2_p0_s0.7
show cartoon, occ_463_c2_p0_s0.7 and chain B+E
color palegreen, occ_463_c2_p0_s0.7 and chain B
color lightblue, occ_463_c2_p0_s0.7 and chain E
select hotspot_source, occ_463_c2_p0_s0.7 and ((chain B and resi 484))
select hotspot_target, occ_463_c2_p0_s0.7 and ((chain E and resi 31))
select hotspot_all, occ_463_c2_p0_s0.7 and ((chain B and resi 484) or (chain E and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_463_c2_p0_s0.7 and chain B+E
set_name hotspot_all, hotspot_occurrence_463
set_name hotspot_source, hotspot_source_463
set_name hotspot_target, hotspot_target_463
bg_color white
# patternId=0 support=0.7 graphId=191
