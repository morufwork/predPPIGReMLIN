load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wnm.ent", occ_115_c0_p0_s0.8
hide everything, occ_115_c0_p0_s0.8
show cartoon, occ_115_c0_p0_s0.8 and chain B+A
color palegreen, occ_115_c0_p0_s0.8 and chain B
color lightblue, occ_115_c0_p0_s0.8 and chain A
select hotspot_source, occ_115_c0_p0_s0.8 and ((chain B and resi 353))
select hotspot_target, occ_115_c0_p0_s0.8 and ((chain A and resi 505))
select hotspot_all, occ_115_c0_p0_s0.8 and ((chain A and resi 505) or (chain B and resi 353))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_115_c0_p0_s0.8 and chain B+A
set_name hotspot_all, hotspot_occurrence_115
set_name hotspot_source, hotspot_source_115
set_name hotspot_target, hotspot_target_115
bg_color white
# patternId=0 support=0.8 graphId=284
