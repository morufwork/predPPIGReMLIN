load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sy0.ent", occ_1300_c3_p0_s0.7
hide everything, occ_1300_c3_p0_s0.7
show cartoon, occ_1300_c3_p0_s0.7 and chain B+E
color palegreen, occ_1300_c3_p0_s0.7 and chain B
color lightblue, occ_1300_c3_p0_s0.7 and chain E
select hotspot_source, occ_1300_c3_p0_s0.7 and ((chain B and resi 502))
select hotspot_target, occ_1300_c3_p0_s0.7 and ((chain E and resi 353))
select hotspot_all, occ_1300_c3_p0_s0.7 and ((chain B and resi 502) or (chain E and resi 353))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1300_c3_p0_s0.7 and chain B+E
set_name hotspot_all, hotspot_occurrence_1300
set_name hotspot_source, hotspot_source_1300
set_name hotspot_target, hotspot_target_1300
bg_color white
# patternId=0 support=0.7 graphId=208
