load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dm6.ent", occ_401_c1_p0_s1.0
hide everything, occ_401_c1_p0_s1.0
show cartoon, occ_401_c1_p0_s1.0 and chain A+D
color palegreen, occ_401_c1_p0_s1.0 and chain A
color lightblue, occ_401_c1_p0_s1.0 and chain D
select hotspot_source, occ_401_c1_p0_s1.0 and ((chain A and resi 500))
select hotspot_target, occ_401_c1_p0_s1.0 and ((chain D and resi 41))
select hotspot_all, occ_401_c1_p0_s1.0 and ((chain A and resi 500) or (chain D and resi 41))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_401_c1_p0_s1.0 and chain A+D
set_name hotspot_all, hotspot_occurrence_401
set_name hotspot_source, hotspot_source_401
set_name hotspot_target, hotspot_target_401
bg_color white
# patternId=0 support=1.0 graphId=377
