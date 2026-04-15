load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wse.ent", occ_255_c0_p0_s1.0
hide everything, occ_255_c0_p0_s1.0
show cartoon, occ_255_c0_p0_s1.0 and chain A+B
color palegreen, occ_255_c0_p0_s1.0 and chain A
color lightblue, occ_255_c0_p0_s1.0 and chain B
select hotspot_source, occ_255_c0_p0_s1.0 and ((chain A and resi 31))
select hotspot_target, occ_255_c0_p0_s1.0 and ((chain B and resi 489))
select hotspot_all, occ_255_c0_p0_s1.0 and ((chain A and resi 31) or (chain B and resi 489))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_255_c0_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_255
set_name hotspot_source, hotspot_source_255
set_name hotspot_target, hotspot_target_255
bg_color white
# patternId=0 support=1.0 graphId=311
