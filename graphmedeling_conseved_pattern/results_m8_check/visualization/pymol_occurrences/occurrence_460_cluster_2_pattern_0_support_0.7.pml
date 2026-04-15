load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sxy.ent", occ_460_c2_p0_s0.7
hide everything, occ_460_c2_p0_s0.7
show cartoon, occ_460_c2_p0_s0.7 and chain B+E
color palegreen, occ_460_c2_p0_s0.7 and chain B
color lightblue, occ_460_c2_p0_s0.7 and chain E
select hotspot_source, occ_460_c2_p0_s0.7 and ((chain B and resi 403))
select hotspot_target, occ_460_c2_p0_s0.7 and ((chain E and resi 37))
select hotspot_all, occ_460_c2_p0_s0.7 and ((chain B and resi 403) or (chain E and resi 37))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_460_c2_p0_s0.7 and chain B+E
set_name hotspot_all, hotspot_occurrence_460
set_name hotspot_source, hotspot_source_460
set_name hotspot_target, hotspot_target_460
bg_color white
# patternId=0 support=0.7 graphId=188
