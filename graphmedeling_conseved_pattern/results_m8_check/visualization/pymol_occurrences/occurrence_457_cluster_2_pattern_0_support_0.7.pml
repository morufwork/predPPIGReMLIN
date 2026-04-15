load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_457_c2_p0_s0.7
hide everything, occ_457_c2_p0_s0.7
show cartoon, occ_457_c2_p0_s0.7 and chain A+C
color palegreen, occ_457_c2_p0_s0.7 and chain A
color lightblue, occ_457_c2_p0_s0.7 and chain C
select hotspot_source, occ_457_c2_p0_s0.7 and ((chain A and resi 484))
select hotspot_target, occ_457_c2_p0_s0.7 and ((chain C and resi 67))
select hotspot_all, occ_457_c2_p0_s0.7 and ((chain A and resi 484) or (chain C and resi 67))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_457_c2_p0_s0.7 and chain A+C
set_name hotspot_all, hotspot_occurrence_457
set_name hotspot_source, hotspot_source_457
set_name hotspot_target, hotspot_target_457
bg_color white
# patternId=0 support=0.7 graphId=186
