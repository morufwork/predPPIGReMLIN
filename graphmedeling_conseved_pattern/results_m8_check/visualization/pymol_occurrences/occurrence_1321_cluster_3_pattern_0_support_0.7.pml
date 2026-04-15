load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wnm.ent", occ_1321_c3_p0_s0.7
hide everything, occ_1321_c3_p0_s0.7
show cartoon, occ_1321_c3_p0_s0.7 and chain B+A
color palegreen, occ_1321_c3_p0_s0.7 and chain B
color lightblue, occ_1321_c3_p0_s0.7 and chain A
select hotspot_source, occ_1321_c3_p0_s0.7 and ((chain B and resi 19))
select hotspot_target, occ_1321_c3_p0_s0.7 and ((chain A and resi 475))
select hotspot_all, occ_1321_c3_p0_s0.7 and ((chain A and resi 475) or (chain B and resi 19))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1321_c3_p0_s0.7 and chain B+A
set_name hotspot_all, hotspot_occurrence_1321
set_name hotspot_source, hotspot_source_1321
set_name hotspot_target, hotspot_target_1321
bg_color white
# patternId=0 support=0.7 graphId=273
