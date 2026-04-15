load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7sxy.ent", occ_664_c2_p0_s0.8
hide everything, occ_664_c2_p0_s0.8
show cartoon, occ_664_c2_p0_s0.8 and chain B+E
color palegreen, occ_664_c2_p0_s0.8 and chain B
color lightblue, occ_664_c2_p0_s0.8 and chain E
select hotspot_source, occ_664_c2_p0_s0.8 and ((chain B and resi 403))
select hotspot_target, occ_664_c2_p0_s0.8 and ((chain E and resi 37))
select hotspot_all, occ_664_c2_p0_s0.8 and ((chain B and resi 403) or (chain E and resi 37))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_664_c2_p0_s0.8 and chain B+E
set_name hotspot_all, hotspot_occurrence_664
set_name hotspot_source, hotspot_source_664
set_name hotspot_target, hotspot_target_664
bg_color white
# patternId=0 support=0.8 graphId=188
