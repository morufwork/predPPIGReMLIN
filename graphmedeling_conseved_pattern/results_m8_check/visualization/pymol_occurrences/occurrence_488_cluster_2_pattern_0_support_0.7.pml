load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w8s.ent", occ_488_c2_p0_s0.7
hide everything, occ_488_c2_p0_s0.7
show cartoon, occ_488_c2_p0_s0.7 and chain A+B
color palegreen, occ_488_c2_p0_s0.7 and chain A
color lightblue, occ_488_c2_p0_s0.7 and chain B
select hotspot_source, occ_488_c2_p0_s0.7 and ((chain A and resi 354))
select hotspot_target, occ_488_c2_p0_s0.7 and ((chain B and resi 405))
select hotspot_all, occ_488_c2_p0_s0.7 and ((chain A and resi 354) or (chain B and resi 405))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_488_c2_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_488
set_name hotspot_source, hotspot_source_488
set_name hotspot_target, hotspot_target_488
bg_color white
# patternId=0 support=0.7 graphId=248
