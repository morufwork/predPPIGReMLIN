load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7fc5.ent", occ_446_c2_p0_s0.7
hide everything, occ_446_c2_p0_s0.7
show cartoon, occ_446_c2_p0_s0.7 and chain E+A
color palegreen, occ_446_c2_p0_s0.7 and chain E
color lightblue, occ_446_c2_p0_s0.7 and chain A
select hotspot_source, occ_446_c2_p0_s0.7 and ((chain E and resi 403))
select hotspot_target, occ_446_c2_p0_s0.7 and ((chain A and resi 37))
select hotspot_all, occ_446_c2_p0_s0.7 and ((chain A and resi 37) or (chain E and resi 403))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_446_c2_p0_s0.7 and chain E+A
set_name hotspot_all, hotspot_occurrence_446
set_name hotspot_source, hotspot_source_446
set_name hotspot_target, hotspot_target_446
bg_color white
# patternId=0 support=0.7 graphId=144
