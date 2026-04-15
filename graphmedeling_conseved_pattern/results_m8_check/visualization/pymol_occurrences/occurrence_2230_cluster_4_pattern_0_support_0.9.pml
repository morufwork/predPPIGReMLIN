load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7e3j.ent", occ_2230_c4_p0_s0.9
hide everything, occ_2230_c4_p0_s0.9
show cartoon, occ_2230_c4_p0_s0.9 and chain A+B
color palegreen, occ_2230_c4_p0_s0.9 and chain A
color lightblue, occ_2230_c4_p0_s0.9 and chain B
select hotspot_source, occ_2230_c4_p0_s0.9 and ((chain A and resi 27))
select hotspot_target, occ_2230_c4_p0_s0.9 and ((chain B and resi 456))
select hotspot_all, occ_2230_c4_p0_s0.9 and ((chain A and resi 27) or (chain B and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2230_c4_p0_s0.9 and chain A+B
set_name hotspot_all, hotspot_occurrence_2230
set_name hotspot_source, hotspot_source_2230
set_name hotspot_target, hotspot_target_2230
bg_color white
# patternId=0 support=0.9 graphId=43
