load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wpb.ent", occ_510_c2_p0_s0.7
hide everything, occ_510_c2_p0_s0.7
show cartoon, occ_510_c2_p0_s0.7 and chain A+D
color palegreen, occ_510_c2_p0_s0.7 and chain A
color lightblue, occ_510_c2_p0_s0.7 and chain D
select hotspot_source, occ_510_c2_p0_s0.7 and ((chain A and resi 490))
select hotspot_target, occ_510_c2_p0_s0.7 and ((chain D and resi 35))
select hotspot_all, occ_510_c2_p0_s0.7 and ((chain A and resi 490) or (chain D and resi 35))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_510_c2_p0_s0.7 and chain A+D
set_name hotspot_all, hotspot_occurrence_510
set_name hotspot_source, hotspot_source_510
set_name hotspot_target, hotspot_target_510
bg_color white
# patternId=0 support=0.7 graphId=303
