load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6m0j.ent", occ_2667_c5_p0_s0.7
hide everything, occ_2667_c5_p0_s0.7
show cartoon, occ_2667_c5_p0_s0.7 and chain A+E
color palegreen, occ_2667_c5_p0_s0.7 and chain A
color lightblue, occ_2667_c5_p0_s0.7 and chain E
select hotspot_source, occ_2667_c5_p0_s0.7 and ((chain A and resi 27))
select hotspot_target, occ_2667_c5_p0_s0.7 and ((chain E and resi 456) or (chain E and resi 489))
select hotspot_all, occ_2667_c5_p0_s0.7 and ((chain A and resi 27) or (chain E and resi 456) or (chain E and resi 489))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2667_c5_p0_s0.7 and chain A+E
set_name hotspot_all, hotspot_occurrence_2667
set_name hotspot_source, hotspot_source_2667
set_name hotspot_target, hotspot_target_2667
bg_color white
# patternId=0 support=0.7 graphId=14
