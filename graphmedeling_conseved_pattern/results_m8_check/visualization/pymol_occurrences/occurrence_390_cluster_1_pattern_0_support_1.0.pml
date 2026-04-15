load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7t9l.ent", occ_390_c1_p0_s1.0
hide everything, occ_390_c1_p0_s1.0
show cartoon, occ_390_c1_p0_s1.0 and chain A+D
color palegreen, occ_390_c1_p0_s1.0 and chain A
color lightblue, occ_390_c1_p0_s1.0 and chain D
select hotspot_source, occ_390_c1_p0_s1.0 and ((chain A and resi 500))
select hotspot_target, occ_390_c1_p0_s1.0 and ((chain D and resi 41))
select hotspot_all, occ_390_c1_p0_s1.0 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_390_c1_p0_s1.0 and chain A+D
set_name hotspot_all, hotspot_occurrence_390
set_name hotspot_source, hotspot_source_390
set_name hotspot_target, hotspot_target_390
bg_color white
# patternId=0 support=1.0 graphId=225
