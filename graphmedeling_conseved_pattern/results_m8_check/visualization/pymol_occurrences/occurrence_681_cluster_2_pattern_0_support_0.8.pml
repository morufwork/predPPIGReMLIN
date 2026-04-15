load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7t9l.ent", occ_681_c2_p0_s0.8
hide everything, occ_681_c2_p0_s0.8
show cartoon, occ_681_c2_p0_s0.8 and chain A+D
color palegreen, occ_681_c2_p0_s0.8 and chain A
color lightblue, occ_681_c2_p0_s0.8 and chain D
select hotspot_source, occ_681_c2_p0_s0.8 and ((chain A and resi 498))
select hotspot_target, occ_681_c2_p0_s0.8 and ((chain D and resi 38))
select hotspot_all, occ_681_c2_p0_s0.8 and ((chain A and resi 498) or (chain D and resi 38))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_681_c2_p0_s0.8 and chain A+D
set_name hotspot_all, hotspot_occurrence_681
set_name hotspot_source, hotspot_source_681
set_name hotspot_target, hotspot_target_681
bg_color white
# patternId=0 support=0.8 graphId=224
