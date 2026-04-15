load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_2758_c5_p0_s0.8
hide everything, occ_2758_c5_p0_s0.8
show cartoon, occ_2758_c5_p0_s0.8 and chain E+A
color palegreen, occ_2758_c5_p0_s0.8 and chain E
color lightblue, occ_2758_c5_p0_s0.8 and chain A
select hotspot_source, occ_2758_c5_p0_s0.8 and ((chain E and resi 505))
select hotspot_target, occ_2758_c5_p0_s0.8 and ((chain A and resi 353))
select hotspot_all, occ_2758_c5_p0_s0.8 and ((chain A and resi 353) or (chain E and resi 505))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2758_c5_p0_s0.8 and chain E+A
set_name hotspot_all, hotspot_occurrence_2758
set_name hotspot_source, hotspot_source_2758
set_name hotspot_target, hotspot_target_2758
bg_color white
# patternId=0 support=0.8 graphId=153
