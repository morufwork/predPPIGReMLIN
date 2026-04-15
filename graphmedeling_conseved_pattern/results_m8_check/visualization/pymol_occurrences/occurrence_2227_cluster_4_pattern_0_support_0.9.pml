load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7dhx.ent", occ_2227_c4_p0_s0.9
hide everything, occ_2227_c4_p0_s0.9
show cartoon, occ_2227_c4_p0_s0.9 and chain A+B
color palegreen, occ_2227_c4_p0_s0.9 and chain A
color lightblue, occ_2227_c4_p0_s0.9 and chain B
select hotspot_source, occ_2227_c4_p0_s0.9 and ((chain A and resi 353))
select hotspot_target, occ_2227_c4_p0_s0.9 and ((chain B and resi 505))
select hotspot_all, occ_2227_c4_p0_s0.9 and ((chain A and resi 353) or (chain B and resi 505))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2227_c4_p0_s0.9 and chain A+B
set_name hotspot_all, hotspot_occurrence_2227
set_name hotspot_source, hotspot_source_2227
set_name hotspot_target, hotspot_target_2227
bg_color white
# patternId=0 support=0.9 graphId=40
