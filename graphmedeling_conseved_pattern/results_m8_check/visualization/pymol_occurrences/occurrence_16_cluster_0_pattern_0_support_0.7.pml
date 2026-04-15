load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7ekc.ent", occ_16_c0_p0_s0.7
hide everything, occ_16_c0_p0_s0.7
show cartoon, occ_16_c0_p0_s0.7 and chain A+B
color palegreen, occ_16_c0_p0_s0.7 and chain A
color lightblue, occ_16_c0_p0_s0.7 and chain B
select hotspot_source, occ_16_c0_p0_s0.7 and ((chain A and resi 27))
select hotspot_target, occ_16_c0_p0_s0.7 and ((chain B and resi 456))
select hotspot_all, occ_16_c0_p0_s0.7 and ((chain A and resi 27) or (chain B and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_16_c0_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_16
set_name hotspot_source, hotspot_source_16
set_name hotspot_target, hotspot_target_16
bg_color white
# patternId=0 support=0.7 graphId=87
