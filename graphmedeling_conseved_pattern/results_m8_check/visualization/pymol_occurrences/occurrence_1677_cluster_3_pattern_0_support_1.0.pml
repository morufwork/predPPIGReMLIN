load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7ekh.ent", occ_1677_c3_p0_s1.0
hide everything, occ_1677_c3_p0_s1.0
show cartoon, occ_1677_c3_p0_s1.0 and chain A+B
color palegreen, occ_1677_c3_p0_s1.0 and chain A
color lightblue, occ_1677_c3_p0_s1.0 and chain B
select hotspot_source, occ_1677_c3_p0_s1.0 and ((chain A and resi 353))
select hotspot_target, occ_1677_c3_p0_s1.0 and ((chain B and resi 502))
select hotspot_all, occ_1677_c3_p0_s1.0 and ((chain A and resi 353) or (chain B and resi 502))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1677_c3_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_1677
set_name hotspot_source, hotspot_source_1677
set_name hotspot_target, hotspot_target_1677
bg_color white
# patternId=0 support=1.0 graphId=141
