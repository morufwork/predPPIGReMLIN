load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wse.ent", occ_520_c2_p0_s0.7
hide everything, occ_520_c2_p0_s0.7
show cartoon, occ_520_c2_p0_s0.7 and chain A+B
color palegreen, occ_520_c2_p0_s0.7 and chain A
color lightblue, occ_520_c2_p0_s0.7 and chain B
select hotspot_source, occ_520_c2_p0_s0.7 and ((chain A and resi 37))
select hotspot_target, occ_520_c2_p0_s0.7 and ((chain B and resi 403))
select hotspot_all, occ_520_c2_p0_s0.7 and ((chain A and resi 37) or (chain B and resi 403))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_520_c2_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_520
set_name hotspot_source, hotspot_source_520
set_name hotspot_target, hotspot_target_520
bg_color white
# patternId=0 support=0.7 graphId=315
