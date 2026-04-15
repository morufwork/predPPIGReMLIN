load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dlq.ent", occ_63_c0_p0_s0.7
hide everything, occ_63_c0_p0_s0.7
show cartoon, occ_63_c0_p0_s0.7 and chain B+E
color palegreen, occ_63_c0_p0_s0.7 and chain B
color lightblue, occ_63_c0_p0_s0.7 and chain E
select hotspot_source, occ_63_c0_p0_s0.7 and ((chain B and resi 486))
select hotspot_target, occ_63_c0_p0_s0.7 and ((chain E and resi 83))
select hotspot_all, occ_63_c0_p0_s0.7 and ((chain B and resi 486) or (chain E and resi 83))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_63_c0_p0_s0.7 and chain B+E
set_name hotspot_all, hotspot_occurrence_63
set_name hotspot_source, hotspot_source_63
set_name hotspot_target, hotspot_target_63
bg_color white
# patternId=0 support=0.7 graphId=366
