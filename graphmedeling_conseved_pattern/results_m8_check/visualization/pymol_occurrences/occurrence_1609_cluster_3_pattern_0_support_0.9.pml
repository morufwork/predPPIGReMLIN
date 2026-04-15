load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xo6.ent", occ_1609_c3_p0_s0.9
hide everything, occ_1609_c3_p0_s0.9
show cartoon, occ_1609_c3_p0_s0.9 and chain D+A
color palegreen, occ_1609_c3_p0_s0.9 and chain D
color lightblue, occ_1609_c3_p0_s0.9 and chain A
select hotspot_source, occ_1609_c3_p0_s0.9 and ((chain D and resi 41))
select hotspot_target, occ_1609_c3_p0_s0.9 and ((chain A and resi 500))
select hotspot_all, occ_1609_c3_p0_s0.9 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1609_c3_p0_s0.9 and chain D+A
set_name hotspot_all, hotspot_occurrence_1609
set_name hotspot_source, hotspot_source_1609
set_name hotspot_target, hotspot_target_1609
bg_color white
# patternId=0 support=0.9 graphId=339
