load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7t9l.ent", occ_480_c2_p0_s0.7
hide everything, occ_480_c2_p0_s0.7
show cartoon, occ_480_c2_p0_s0.7 and chain A+D
color palegreen, occ_480_c2_p0_s0.7 and chain A
color lightblue, occ_480_c2_p0_s0.7 and chain D
select hotspot_source, occ_480_c2_p0_s0.7 and ((chain A and resi 498))
select hotspot_target, occ_480_c2_p0_s0.7 and ((chain D and resi 38))
select hotspot_all, occ_480_c2_p0_s0.7 and ((chain A and resi 498) or (chain D and resi 38))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_480_c2_p0_s0.7 and chain A+D
set_name hotspot_all, hotspot_occurrence_480
set_name hotspot_source, hotspot_source_480
set_name hotspot_target, hotspot_target_480
bg_color white
# patternId=0 support=0.7 graphId=224
