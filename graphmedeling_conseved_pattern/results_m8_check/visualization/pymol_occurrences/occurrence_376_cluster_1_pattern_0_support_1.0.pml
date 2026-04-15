load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7efr.ent", occ_376_c1_p0_s1.0
hide everything, occ_376_c1_p0_s1.0
show cartoon, occ_376_c1_p0_s1.0 and chain A+B
color palegreen, occ_376_c1_p0_s1.0 and chain A
color lightblue, occ_376_c1_p0_s1.0 and chain B
select hotspot_source, occ_376_c1_p0_s1.0 and ((chain A and resi 41))
select hotspot_target, occ_376_c1_p0_s1.0 and ((chain B and resi 500))
select hotspot_all, occ_376_c1_p0_s1.0 and ((chain A and resi 41) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_376_c1_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_376
set_name hotspot_source, hotspot_source_376
set_name hotspot_target, hotspot_target_376
bg_color white
# patternId=0 support=1.0 graphId=76
