load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sxy.ent", occ_1563_c3_p0_s0.9
hide everything, occ_1563_c3_p0_s0.9
show cartoon, occ_1563_c3_p0_s0.9 and chain B+E
color palegreen, occ_1563_c3_p0_s0.9 and chain B
color lightblue, occ_1563_c3_p0_s0.9 and chain E
select hotspot_source, occ_1563_c3_p0_s0.9 and ((chain B and resi 449))
select hotspot_target, occ_1563_c3_p0_s0.9 and ((chain E and resi 38))
select hotspot_all, occ_1563_c3_p0_s0.9 and ((chain B and resi 449) or (chain E and resi 38))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1563_c3_p0_s0.9 and chain B+E
set_name hotspot_all, hotspot_occurrence_1563
set_name hotspot_source, hotspot_source_1563
set_name hotspot_target, hotspot_target_1563
bg_color white
# patternId=0 support=0.9 graphId=190
