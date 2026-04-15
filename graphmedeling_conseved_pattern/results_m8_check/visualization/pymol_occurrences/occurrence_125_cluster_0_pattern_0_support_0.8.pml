load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wsh.ent", occ_125_c0_p0_s0.8
hide everything, occ_125_c0_p0_s0.8
show cartoon, occ_125_c0_p0_s0.8 and chain A+B
color palegreen, occ_125_c0_p0_s0.8 and chain A
color lightblue, occ_125_c0_p0_s0.8 and chain B
select hotspot_source, occ_125_c0_p0_s0.8 and ((chain A and resi 38))
select hotspot_target, occ_125_c0_p0_s0.8 and ((chain B and resi 449))
select hotspot_all, occ_125_c0_p0_s0.8 and ((chain A and resi 38) or (chain B and resi 449))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_125_c0_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_125
set_name hotspot_source, hotspot_source_125
set_name hotspot_target, hotspot_target_125
bg_color white
# patternId=0 support=0.8 graphId=328
