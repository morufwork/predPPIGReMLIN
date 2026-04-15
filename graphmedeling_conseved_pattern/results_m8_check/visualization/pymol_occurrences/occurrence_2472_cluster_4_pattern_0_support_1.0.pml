load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7ekg.ent", occ_2472_c4_p0_s1.0
hide everything, occ_2472_c4_p0_s1.0
show cartoon, occ_2472_c4_p0_s1.0 and chain A+B
color palegreen, occ_2472_c4_p0_s1.0 and chain A
color lightblue, occ_2472_c4_p0_s1.0 and chain B
select hotspot_source, occ_2472_c4_p0_s1.0 and ((chain A and resi 41) or (chain A and resi 353))
select hotspot_target, occ_2472_c4_p0_s1.0 and ((chain B and resi 501))
select hotspot_all, occ_2472_c4_p0_s1.0 and ((chain A and resi 41) or (chain A and resi 353) or (chain B and resi 501))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2472_c4_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_2472
set_name hotspot_source, hotspot_source_2472
set_name hotspot_target, hotspot_target_2472
bg_color white
# patternId=0 support=1.0 graphId=124
