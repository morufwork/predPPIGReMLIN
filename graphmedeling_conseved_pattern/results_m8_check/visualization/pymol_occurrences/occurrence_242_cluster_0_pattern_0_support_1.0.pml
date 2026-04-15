load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wa1.ent", occ_242_c0_p0_s1.0
hide everything, occ_242_c0_p0_s1.0
show cartoon, occ_242_c0_p0_s1.0 and chain A+B
color palegreen, occ_242_c0_p0_s1.0 and chain A
color lightblue, occ_242_c0_p0_s1.0 and chain B
select hotspot_source, occ_242_c0_p0_s1.0 and ((chain A and resi 24))
select hotspot_target, occ_242_c0_p0_s1.0 and ((chain B and resi 475))
select hotspot_all, occ_242_c0_p0_s1.0 and ((chain A and resi 24) or (chain B and resi 475))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_242_c0_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_242
set_name hotspot_source, hotspot_source_242
set_name hotspot_target, hotspot_target_242
bg_color white
# patternId=0 support=1.0 graphId=259
