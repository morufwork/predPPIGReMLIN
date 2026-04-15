load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", rep_c2_p0_s0.7
hide everything, rep_c2_p0_s0.7
show cartoon, rep_c2_p0_s0.7 and chain A+B
color palegreen, rep_c2_p0_s0.7 and chain A
color lightblue, rep_c2_p0_s0.7 and chain B
select hotspot_source, rep_c2_p0_s0.7 and ((chain A and resi 31))
select hotspot_target, rep_c2_p0_s0.7 and ((chain B and resi 484))
select hotspot_all, rep_c2_p0_s0.7 and ((chain A and resi 31) or (chain B and resi 484))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient rep_c2_p0_s0.7 and chain A+B
bg_color white
set_name hotspot_all, representative_hotspot_2_0
set_name hotspot_source, representative_source_2_0
set_name hotspot_target, representative_target_2_0
# representative occurrenceId=404 graphId=4
