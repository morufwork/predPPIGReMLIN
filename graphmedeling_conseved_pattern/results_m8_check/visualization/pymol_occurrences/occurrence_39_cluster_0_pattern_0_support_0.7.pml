load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w8s.ent", occ_39_c0_p0_s0.7
hide everything, occ_39_c0_p0_s0.7
show cartoon, occ_39_c0_p0_s0.7 and chain A+B
color palegreen, occ_39_c0_p0_s0.7 and chain A
color lightblue, occ_39_c0_p0_s0.7 and chain B
select hotspot_source, occ_39_c0_p0_s0.7 and ((chain A and resi 27))
select hotspot_target, occ_39_c0_p0_s0.7 and ((chain B and resi 456))
select hotspot_all, occ_39_c0_p0_s0.7 and ((chain A and resi 27) or (chain B and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_39_c0_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_39
set_name hotspot_source, hotspot_source_39
set_name hotspot_target, hotspot_target_39
bg_color white
# patternId=0 support=0.7 graphId=243
