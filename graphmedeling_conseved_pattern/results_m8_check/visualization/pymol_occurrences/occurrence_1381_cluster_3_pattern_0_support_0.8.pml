load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7efr.ent", occ_1381_c3_p0_s0.8
hide everything, occ_1381_c3_p0_s0.8
show cartoon, occ_1381_c3_p0_s0.8 and chain A+B
color palegreen, occ_1381_c3_p0_s0.8 and chain A
color lightblue, occ_1381_c3_p0_s0.8 and chain B
select hotspot_source, occ_1381_c3_p0_s0.8 and ((chain A and resi 42))
select hotspot_target, occ_1381_c3_p0_s0.8 and ((chain B and resi 498))
select hotspot_all, occ_1381_c3_p0_s0.8 and ((chain A and resi 42) or (chain B and resi 498))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1381_c3_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_1381
set_name hotspot_source, hotspot_source_1381
set_name hotspot_target, hotspot_target_1381
bg_color white
# patternId=0 support=0.8 graphId=77
