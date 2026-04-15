load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sxy.ent", occ_871_c2_p0_s0.9
hide everything, occ_871_c2_p0_s0.9
show cartoon, occ_871_c2_p0_s0.9 and chain B+E
color palegreen, occ_871_c2_p0_s0.9 and chain B
color lightblue, occ_871_c2_p0_s0.9 and chain E
select hotspot_source, occ_871_c2_p0_s0.9 and ((chain B and resi 484))
select hotspot_target, occ_871_c2_p0_s0.9 and ((chain E and resi 31))
select hotspot_all, occ_871_c2_p0_s0.9 and ((chain B and resi 484) or (chain E and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_871_c2_p0_s0.9 and chain B+E
set_name hotspot_all, hotspot_occurrence_871
set_name hotspot_source, hotspot_source_871
set_name hotspot_target, hotspot_target_871
bg_color white
# patternId=0 support=0.9 graphId=191
