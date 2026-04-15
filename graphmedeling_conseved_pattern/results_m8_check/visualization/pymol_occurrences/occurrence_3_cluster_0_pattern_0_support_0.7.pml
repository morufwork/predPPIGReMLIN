load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7bh9.ent", occ_3_c0_p0_s0.7
hide everything, occ_3_c0_p0_s0.7
show cartoon, occ_3_c0_p0_s0.7 and chain A+E
color palegreen, occ_3_c0_p0_s0.7 and chain A
color lightblue, occ_3_c0_p0_s0.7 and chain E
select hotspot_source, occ_3_c0_p0_s0.7 and ((chain A and resi 353))
select hotspot_target, occ_3_c0_p0_s0.7 and ((chain E and resi 501))
select hotspot_all, occ_3_c0_p0_s0.7 and ((chain A and resi 353) or (chain E and resi 501))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_3_c0_p0_s0.7 and chain A+E
set_name hotspot_all, hotspot_occurrence_3
set_name hotspot_source, hotspot_source_3
set_name hotspot_target, hotspot_target_3
bg_color white
# patternId=0 support=0.7 graphId=31
