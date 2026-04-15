load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dlq.ent", occ_1754_c3_p0_s1.0
hide everything, occ_1754_c3_p0_s1.0
show cartoon, occ_1754_c3_p0_s1.0 and chain B+E
color palegreen, occ_1754_c3_p0_s1.0 and chain B
color lightblue, occ_1754_c3_p0_s1.0 and chain E
select hotspot_source, occ_1754_c3_p0_s1.0 and ((chain B and resi 493))
select hotspot_target, occ_1754_c3_p0_s1.0 and ((chain E and resi 34))
select hotspot_all, occ_1754_c3_p0_s1.0 and ((chain B and resi 493) or (chain E and resi 34))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1754_c3_p0_s1.0 and chain B+E
set_name hotspot_all, hotspot_occurrence_1754
set_name hotspot_source, hotspot_source_1754
set_name hotspot_target, hotspot_target_1754
bg_color white
# patternId=0 support=1.0 graphId=368
