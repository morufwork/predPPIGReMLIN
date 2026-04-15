load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7spp.ent", occ_98_c0_p0_s0.8
hide everything, occ_98_c0_p0_s0.8
show cartoon, occ_98_c0_p0_s0.8 and chain A+C
color palegreen, occ_98_c0_p0_s0.8 and chain A
color lightblue, occ_98_c0_p0_s0.8 and chain C
select hotspot_source, occ_98_c0_p0_s0.8 and ((chain A and resi 354))
select hotspot_target, occ_98_c0_p0_s0.8 and ((chain C and resi 114))
select hotspot_all, occ_98_c0_p0_s0.8 and ((chain A and resi 354) or (chain C and resi 114))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_98_c0_p0_s0.8 and chain A+C
set_name hotspot_all, hotspot_occurrence_98
set_name hotspot_source, hotspot_source_98
set_name hotspot_target, hotspot_target_98
bg_color white
# patternId=0 support=0.8 graphId=182
