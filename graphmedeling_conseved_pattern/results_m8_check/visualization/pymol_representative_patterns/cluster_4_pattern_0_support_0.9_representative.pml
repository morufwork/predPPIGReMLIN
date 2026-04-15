load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7dhx.ent", rep_c4_p0_s0.9
hide everything, rep_c4_p0_s0.9
show cartoon, rep_c4_p0_s0.9 and chain A+B
color palegreen, rep_c4_p0_s0.9 and chain A
color lightblue, rep_c4_p0_s0.9 and chain B
select hotspot_source, rep_c4_p0_s0.9 and ((chain A and resi 27))
select hotspot_target, rep_c4_p0_s0.9 and ((chain B and resi 456) or (chain B and resi 475))
select hotspot_all, rep_c4_p0_s0.9 and ((chain A and resi 27) or (chain B and resi 456) or (chain B and resi 475))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient rep_c4_p0_s0.9 and chain A+B
bg_color white
set_name hotspot_all, representative_hotspot_4_0
set_name hotspot_source, representative_source_4_0
set_name hotspot_target, representative_target_4_0
# representative occurrenceId=2212 graphId=33
