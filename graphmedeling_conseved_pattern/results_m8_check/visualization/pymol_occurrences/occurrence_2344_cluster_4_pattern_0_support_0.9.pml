load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wsh.ent", occ_2344_c4_p0_s0.9
hide everything, occ_2344_c4_p0_s0.9
show cartoon, occ_2344_c4_p0_s0.9 and chain A+B
color palegreen, occ_2344_c4_p0_s0.9 and chain A
color lightblue, occ_2344_c4_p0_s0.9 and chain B
select hotspot_source, occ_2344_c4_p0_s0.9 and ((chain A and resi 31))
select hotspot_target, occ_2344_c4_p0_s0.9 and ((chain B and resi 456))
select hotspot_all, occ_2344_c4_p0_s0.9 and ((chain A and resi 31) or (chain B and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2344_c4_p0_s0.9 and chain A+B
set_name hotspot_all, hotspot_occurrence_2344
set_name hotspot_source, hotspot_source_2344
set_name hotspot_target, hotspot_target_2344
bg_color white
# patternId=0 support=0.9 graphId=326
