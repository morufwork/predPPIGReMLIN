load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w9i.ent", occ_392_c1_p0_s1.0
hide everything, occ_392_c1_p0_s1.0
show cartoon, occ_392_c1_p0_s1.0 and chain A+E
color palegreen, occ_392_c1_p0_s1.0 and chain A
color lightblue, occ_392_c1_p0_s1.0 and chain E
select hotspot_source, occ_392_c1_p0_s1.0 and ((chain A and resi 41))
select hotspot_target, occ_392_c1_p0_s1.0 and ((chain E and resi 500))
select hotspot_all, occ_392_c1_p0_s1.0 and ((chain A and resi 41) or (chain E and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_392_c1_p0_s1.0 and chain A+E
set_name hotspot_all, hotspot_occurrence_392
set_name hotspot_source, hotspot_source_392
set_name hotspot_target, hotspot_target_392
bg_color white
# patternId=0 support=1.0 graphId=255
