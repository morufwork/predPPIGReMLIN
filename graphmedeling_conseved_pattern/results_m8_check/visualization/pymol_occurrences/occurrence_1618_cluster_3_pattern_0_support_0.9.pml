load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dlq.ent", occ_1618_c3_p0_s0.9
hide everything, occ_1618_c3_p0_s0.9
show cartoon, occ_1618_c3_p0_s0.9 and chain B+E
color palegreen, occ_1618_c3_p0_s0.9 and chain B
color lightblue, occ_1618_c3_p0_s0.9 and chain E
select hotspot_source, occ_1618_c3_p0_s0.9 and ((chain B and resi 487))
select hotspot_target, occ_1618_c3_p0_s0.9 and ((chain E and resi 83))
select hotspot_all, occ_1618_c3_p0_s0.9 and ((chain B and resi 487) or (chain E and resi 83))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1618_c3_p0_s0.9 and chain B+E
set_name hotspot_all, hotspot_occurrence_1618
set_name hotspot_source, hotspot_source_1618
set_name hotspot_target, hotspot_target_1618
bg_color white
# patternId=0 support=0.9 graphId=367
