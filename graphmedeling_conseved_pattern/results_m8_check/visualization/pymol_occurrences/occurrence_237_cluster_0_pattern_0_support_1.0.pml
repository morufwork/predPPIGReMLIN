load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7t9l.ent", occ_237_c0_p0_s1.0
hide everything, occ_237_c0_p0_s1.0
show cartoon, occ_237_c0_p0_s1.0 and chain A+D
color palegreen, occ_237_c0_p0_s1.0 and chain A
color lightblue, occ_237_c0_p0_s1.0 and chain D
select hotspot_source, occ_237_c0_p0_s1.0 and ((chain A and resi 489))
select hotspot_target, occ_237_c0_p0_s1.0 and ((chain D and resi 31))
select hotspot_all, occ_237_c0_p0_s1.0 and ((chain A and resi 489) or (chain D and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_237_c0_p0_s1.0 and chain A+D
set_name hotspot_all, hotspot_occurrence_237
set_name hotspot_source, hotspot_source_237
set_name hotspot_target, hotspot_target_237
bg_color white
# patternId=0 support=1.0 graphId=222
