load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wnm.ent", occ_46_c0_p0_s0.7
hide everything, occ_46_c0_p0_s0.7
show cartoon, occ_46_c0_p0_s0.7 and chain B+A
color palegreen, occ_46_c0_p0_s0.7 and chain B
color lightblue, occ_46_c0_p0_s0.7 and chain A
select hotspot_source, occ_46_c0_p0_s0.7 and ((chain B and resi 31))
select hotspot_target, occ_46_c0_p0_s0.7 and ((chain A and resi 489))
select hotspot_all, occ_46_c0_p0_s0.7 and ((chain A and resi 489) or (chain B and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_46_c0_p0_s0.7 and chain B+A
set_name hotspot_all, hotspot_occurrence_46
set_name hotspot_source, hotspot_source_46
set_name hotspot_target, hotspot_target_46
bg_color white
# patternId=0 support=0.7 graphId=276
