load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", rep_c3_p0_s0.8
hide everything, rep_c3_p0_s0.8
show cartoon, rep_c3_p0_s0.8 and chain A+B
color palegreen, rep_c3_p0_s0.8 and chain A
color lightblue, rep_c3_p0_s0.8 and chain B
select hotspot_source, rep_c3_p0_s0.8 and ((chain A and resi 19))
select hotspot_target, rep_c3_p0_s0.8 and ((chain B and resi 475))
select hotspot_all, rep_c3_p0_s0.8 and ((chain A and resi 19) or (chain B and resi 475))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient rep_c3_p0_s0.8 and chain A+B
bg_color white
set_name hotspot_all, representative_hotspot_3_0
set_name hotspot_source, representative_source_3_0
set_name hotspot_target, representative_target_3_0
# representative occurrenceId=1355 graphId=0
