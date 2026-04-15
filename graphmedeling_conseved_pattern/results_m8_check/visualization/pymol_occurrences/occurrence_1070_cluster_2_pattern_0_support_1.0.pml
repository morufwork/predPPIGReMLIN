load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_1070_c2_p0_s1.0
hide everything, occ_1070_c2_p0_s1.0
show cartoon, occ_1070_c2_p0_s1.0 and chain A+C
color palegreen, occ_1070_c2_p0_s1.0 and chain A
color lightblue, occ_1070_c2_p0_s1.0 and chain C
select hotspot_source, occ_1070_c2_p0_s1.0 and ((chain A and resi 484))
select hotspot_target, occ_1070_c2_p0_s1.0 and ((chain C and resi 67))
select hotspot_all, occ_1070_c2_p0_s1.0 and ((chain A and resi 484) or (chain C and resi 67))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1070_c2_p0_s1.0 and chain A+C
set_name hotspot_all, hotspot_occurrence_1070
set_name hotspot_source, hotspot_source_1070
set_name hotspot_target, hotspot_target_1070
bg_color white
# patternId=0 support=1.0 graphId=186
