load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", rep_c1_p0_s0.8
hide everything, rep_c1_p0_s0.8
show cartoon, rep_c1_p0_s0.8 and chain A+B
color palegreen, rep_c1_p0_s0.8 and chain A
color lightblue, rep_c1_p0_s0.8 and chain B
select hotspot_source, rep_c1_p0_s0.8 and ((chain A and resi 41))
select hotspot_target, rep_c1_p0_s0.8 and ((chain B and resi 500))
select hotspot_all, rep_c1_p0_s0.8 and ((chain A and resi 41) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient rep_c1_p0_s0.8 and chain A+B
bg_color white
set_name hotspot_all, representative_hotspot_1_0
set_name hotspot_source, representative_source_1_0
set_name hotspot_target, representative_target_1_0
# representative occurrenceId=302 graphId=6
