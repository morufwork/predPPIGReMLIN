load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dma.ent", occ_2649_c4_p0_s1.0
hide everything, occ_2649_c4_p0_s1.0
show cartoon, occ_2649_c4_p0_s1.0 and chain A+D
color palegreen, occ_2649_c4_p0_s1.0 and chain A
color lightblue, occ_2649_c4_p0_s1.0 and chain D
select hotspot_source, occ_2649_c4_p0_s1.0 and ((chain A and resi 456) or (chain A and resi 489))
select hotspot_target, occ_2649_c4_p0_s1.0 and ((chain D and resi 31))
select hotspot_all, occ_2649_c4_p0_s1.0 and ((chain A and resi 456) or (chain A and resi 489) or (chain D and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2649_c4_p0_s1.0 and chain A+D
set_name hotspot_all, hotspot_occurrence_2649
set_name hotspot_source, hotspot_source_2649
set_name hotspot_target, hotspot_target_2649
bg_color white
# patternId=0 support=1.0 graphId=390
