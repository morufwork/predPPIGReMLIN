load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7bh9.ent", occ_408_c2_p0_s0.7
hide everything, occ_408_c2_p0_s0.7
show cartoon, occ_408_c2_p0_s0.7 and chain A+E
color palegreen, occ_408_c2_p0_s0.7 and chain A
color lightblue, occ_408_c2_p0_s0.7 and chain E
select hotspot_source, occ_408_c2_p0_s0.7 and ((chain A and resi 30))
select hotspot_target, occ_408_c2_p0_s0.7 and ((chain E and resi 417))
select hotspot_all, occ_408_c2_p0_s0.7 and ((chain A and resi 30) or (chain E and resi 417))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_408_c2_p0_s0.7 and chain A+E
set_name hotspot_all, hotspot_occurrence_408
set_name hotspot_source, hotspot_source_408
set_name hotspot_target, hotspot_target_408
bg_color white
# patternId=0 support=0.7 graphId=27
