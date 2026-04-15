load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xo6.ent", occ_1743_c3_p0_s1.0
hide everything, occ_1743_c3_p0_s1.0
show cartoon, occ_1743_c3_p0_s1.0 and chain D+A
color palegreen, occ_1743_c3_p0_s1.0 and chain D
color lightblue, occ_1743_c3_p0_s1.0 and chain A
select hotspot_source, occ_1743_c3_p0_s1.0 and ((chain D and resi 31))
select hotspot_target, occ_1743_c3_p0_s1.0 and ((chain A and resi 493))
select hotspot_all, occ_1743_c3_p0_s1.0 and ((chain A and resi 493) or (chain D and resi 31))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1743_c3_p0_s1.0 and chain D+A
set_name hotspot_all, hotspot_occurrence_1743
set_name hotspot_source, hotspot_source_1743
set_name hotspot_target, hotspot_target_1743
bg_color white
# patternId=0 support=1.0 graphId=336
