load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xoc.ent", occ_365_c1_p0_s0.9
hide everything, occ_365_c1_p0_s0.9
show cartoon, occ_365_c1_p0_s0.9 and chain D+A
color palegreen, occ_365_c1_p0_s0.9 and chain D
color lightblue, occ_365_c1_p0_s0.9 and chain A
select hotspot_source, occ_365_c1_p0_s0.9 and ((chain D and resi 41))
select hotspot_target, occ_365_c1_p0_s0.9 and ((chain A and resi 500))
select hotspot_all, occ_365_c1_p0_s0.9 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_365_c1_p0_s0.9 and chain D+A
set_name hotspot_all, hotspot_occurrence_365
set_name hotspot_source, hotspot_source_365
set_name hotspot_target, hotspot_target_365
bg_color white
# patternId=0 support=0.9 graphId=359
