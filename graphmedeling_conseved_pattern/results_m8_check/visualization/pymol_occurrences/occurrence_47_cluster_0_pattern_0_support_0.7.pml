load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wnm.ent", occ_47_c0_p0_s0.7
hide everything, occ_47_c0_p0_s0.7
show cartoon, occ_47_c0_p0_s0.7 and chain B+A
color palegreen, occ_47_c0_p0_s0.7 and chain B
color lightblue, occ_47_c0_p0_s0.7 and chain A
select hotspot_source, occ_47_c0_p0_s0.7 and ((chain B and resi 83))
select hotspot_target, occ_47_c0_p0_s0.7 and ((chain A and resi 486))
select hotspot_all, occ_47_c0_p0_s0.7 and ((chain A and resi 486) or (chain B and resi 83))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_47_c0_p0_s0.7 and chain B+A
set_name hotspot_all, hotspot_occurrence_47
set_name hotspot_source, hotspot_source_47
set_name hotspot_target, hotspot_target_47
bg_color white
# patternId=0 support=0.7 graphId=281
