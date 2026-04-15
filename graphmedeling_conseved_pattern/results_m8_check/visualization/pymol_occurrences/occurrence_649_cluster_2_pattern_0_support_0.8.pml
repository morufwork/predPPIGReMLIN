load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_649_c2_p0_s0.8
hide everything, occ_649_c2_p0_s0.8
show cartoon, occ_649_c2_p0_s0.8 and chain E+A
color palegreen, occ_649_c2_p0_s0.8 and chain E
color lightblue, occ_649_c2_p0_s0.8 and chain A
select hotspot_source, occ_649_c2_p0_s0.8 and ((chain E and resi 403))
select hotspot_target, occ_649_c2_p0_s0.8 and ((chain A and resi 37))
select hotspot_all, occ_649_c2_p0_s0.8 and ((chain A and resi 37) or (chain E and resi 403))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_649_c2_p0_s0.8 and chain E+A
set_name hotspot_all, hotspot_occurrence_649
set_name hotspot_source, hotspot_source_649
set_name hotspot_target, hotspot_target_649
bg_color white
# patternId=0 support=0.8 graphId=144
