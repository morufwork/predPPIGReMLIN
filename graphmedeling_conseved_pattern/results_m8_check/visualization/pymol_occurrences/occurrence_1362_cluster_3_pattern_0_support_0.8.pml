load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6m0j.ent", occ_1362_c3_p0_s0.8
hide everything, occ_1362_c3_p0_s0.8
show cartoon, occ_1362_c3_p0_s0.8 and chain A+E
color palegreen, occ_1362_c3_p0_s0.8 and chain A
color lightblue, occ_1362_c3_p0_s0.8 and chain E
select hotspot_source, occ_1362_c3_p0_s0.8 and ((chain A and resi 83))
select hotspot_target, occ_1362_c3_p0_s0.8 and ((chain E and resi 487))
select hotspot_all, occ_1362_c3_p0_s0.8 and ((chain A and resi 83) or (chain E and resi 487))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1362_c3_p0_s0.8 and chain A+E
set_name hotspot_all, hotspot_occurrence_1362
set_name hotspot_source, hotspot_source_1362
set_name hotspot_target, hotspot_target_1362
bg_color white
# patternId=0 support=0.8 graphId=22
