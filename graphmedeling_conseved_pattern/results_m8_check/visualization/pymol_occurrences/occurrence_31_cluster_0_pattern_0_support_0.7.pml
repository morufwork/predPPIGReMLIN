load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_31_c0_p0_s0.7
hide everything, occ_31_c0_p0_s0.7
show cartoon, occ_31_c0_p0_s0.7 and chain A+C
color palegreen, occ_31_c0_p0_s0.7 and chain A
color lightblue, occ_31_c0_p0_s0.7 and chain C
select hotspot_source, occ_31_c0_p0_s0.7 and ((chain A and resi 354))
select hotspot_target, occ_31_c0_p0_s0.7 and ((chain C and resi 114))
select hotspot_all, occ_31_c0_p0_s0.7 and ((chain A and resi 354) or (chain C and resi 114))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_31_c0_p0_s0.7 and chain A+C
set_name hotspot_all, hotspot_occurrence_31
set_name hotspot_source, hotspot_source_31
set_name hotspot_target, hotspot_target_31
bg_color white
# patternId=0 support=0.7 graphId=182
