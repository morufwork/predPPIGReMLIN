load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w9i.ent", occ_2537_c4_p0_s1.0
hide everything, occ_2537_c4_p0_s1.0
show cartoon, occ_2537_c4_p0_s1.0 and chain A+E
color palegreen, occ_2537_c4_p0_s1.0 and chain A
color lightblue, occ_2537_c4_p0_s1.0 and chain E
select hotspot_source, occ_2537_c4_p0_s1.0 and ((chain A and resi 27))
select hotspot_target, occ_2537_c4_p0_s1.0 and ((chain E and resi 456) or (chain E and resi 489))
select hotspot_all, occ_2537_c4_p0_s1.0 and ((chain A and resi 27) or (chain E and resi 456) or (chain E and resi 489))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2537_c4_p0_s1.0 and chain A+E
set_name hotspot_all, hotspot_occurrence_2537
set_name hotspot_source, hotspot_source_2537
set_name hotspot_target, hotspot_target_2537
bg_color white
# patternId=0 support=1.0 graphId=250
