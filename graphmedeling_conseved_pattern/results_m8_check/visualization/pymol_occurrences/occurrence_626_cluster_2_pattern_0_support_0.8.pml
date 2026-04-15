load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7e3j.ent", occ_626_c2_p0_s0.8
hide everything, occ_626_c2_p0_s0.8
show cartoon, occ_626_c2_p0_s0.8 and chain A+B
color palegreen, occ_626_c2_p0_s0.8 and chain A
color lightblue, occ_626_c2_p0_s0.8 and chain B
select hotspot_source, occ_626_c2_p0_s0.8 and ((chain A and resi 30))
select hotspot_target, occ_626_c2_p0_s0.8 and ((chain B and resi 417))
select hotspot_all, occ_626_c2_p0_s0.8 and ((chain A and resi 30) or (chain B and resi 417))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_626_c2_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_626
set_name hotspot_source, hotspot_source_626
set_name hotspot_target, hotspot_target_626
bg_color white
# patternId=0 support=0.8 graphId=44
