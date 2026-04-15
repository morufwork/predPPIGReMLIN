load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xoc.ent", occ_399_c1_p0_s1.0
hide everything, occ_399_c1_p0_s1.0
show cartoon, occ_399_c1_p0_s1.0 and chain D+A
color palegreen, occ_399_c1_p0_s1.0 and chain D
color lightblue, occ_399_c1_p0_s1.0 and chain A
select hotspot_source, occ_399_c1_p0_s1.0 and ((chain D and resi 41))
select hotspot_target, occ_399_c1_p0_s1.0 and ((chain A and resi 500))
select hotspot_all, occ_399_c1_p0_s1.0 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_399_c1_p0_s1.0 and chain D+A
set_name hotspot_all, hotspot_occurrence_399
set_name hotspot_source, hotspot_source_399
set_name hotspot_target, hotspot_target_399
bg_color white
# patternId=0 support=1.0 graphId=359
