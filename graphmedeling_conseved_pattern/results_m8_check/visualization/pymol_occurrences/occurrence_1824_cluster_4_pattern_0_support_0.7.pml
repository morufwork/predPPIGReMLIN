load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_1824_c4_p0_s0.7
hide everything, occ_1824_c4_p0_s0.7
show cartoon, occ_1824_c4_p0_s0.7 and chain E+A
color palegreen, occ_1824_c4_p0_s0.7 and chain E
color lightblue, occ_1824_c4_p0_s0.7 and chain A
select hotspot_source, occ_1824_c4_p0_s0.7 and ((chain E and resi 456) or (chain E and resi 489))
select hotspot_target, occ_1824_c4_p0_s0.7 and ((chain A and resi 27))
select hotspot_all, occ_1824_c4_p0_s0.7 and ((chain A and resi 27) or (chain E and resi 456) or (chain E and resi 489))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1824_c4_p0_s0.7 and chain E+A
set_name hotspot_all, hotspot_occurrence_1824
set_name hotspot_source, hotspot_source_1824
set_name hotspot_target, hotspot_target_1824
bg_color white
# patternId=0 support=0.7 graphId=147
