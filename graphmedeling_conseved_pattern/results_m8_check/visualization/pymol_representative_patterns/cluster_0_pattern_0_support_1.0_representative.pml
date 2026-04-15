load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", rep_c0_p0_s1.0
hide everything, rep_c0_p0_s1.0
show cartoon, rep_c0_p0_s1.0 and chain A+B
color palegreen, rep_c0_p0_s1.0 and chain A
color lightblue, rep_c0_p0_s1.0 and chain B
select hotspot_source, rep_c0_p0_s1.0 and ((chain A and resi 83))
select hotspot_target, rep_c0_p0_s1.0 and ((chain B and resi 486))
select hotspot_all, rep_c0_p0_s1.0 and ((chain A and resi 83) or (chain B and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient rep_c0_p0_s1.0 and chain A+B
bg_color white
set_name hotspot_all, representative_hotspot_0_0
set_name hotspot_source, representative_source_0_0
set_name hotspot_target, representative_target_0_0
# representative occurrenceId=201 graphId=9
