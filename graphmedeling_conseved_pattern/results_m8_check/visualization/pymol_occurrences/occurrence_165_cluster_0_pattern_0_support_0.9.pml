load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_165_c0_p0_s0.9
hide everything, occ_165_c0_p0_s0.9
show cartoon, occ_165_c0_p0_s0.9 and chain A+C
color palegreen, occ_165_c0_p0_s0.9 and chain A
color lightblue, occ_165_c0_p0_s0.9 and chain C
select hotspot_source, occ_165_c0_p0_s0.9 and ((chain A and resi 354))
select hotspot_target, occ_165_c0_p0_s0.9 and ((chain C and resi 114))
select hotspot_all, occ_165_c0_p0_s0.9 and ((chain A and resi 354) or (chain C and resi 114))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_165_c0_p0_s0.9 and chain A+C
set_name hotspot_all, hotspot_occurrence_165
set_name hotspot_source, hotspot_source_165
set_name hotspot_target, hotspot_target_165
bg_color white
# patternId=0 support=0.9 graphId=182
