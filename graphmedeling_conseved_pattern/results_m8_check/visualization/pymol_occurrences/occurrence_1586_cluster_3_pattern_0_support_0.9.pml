load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w9i.ent", occ_1586_c3_p0_s0.9
hide everything, occ_1586_c3_p0_s0.9
show cartoon, occ_1586_c3_p0_s0.9 and chain A+E
color palegreen, occ_1586_c3_p0_s0.9 and chain A
color lightblue, occ_1586_c3_p0_s0.9 and chain E
select hotspot_source, occ_1586_c3_p0_s0.9 and ((chain A and resi 42))
select hotspot_target, occ_1586_c3_p0_s0.9 and ((chain E and resi 449))
select hotspot_all, occ_1586_c3_p0_s0.9 and ((chain A and resi 42) or (chain E and resi 449))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1586_c3_p0_s0.9 and chain A+E
set_name hotspot_all, hotspot_occurrence_1586
set_name hotspot_source, hotspot_source_1586
set_name hotspot_target, hotspot_target_1586
bg_color white
# patternId=0 support=0.9 graphId=256
