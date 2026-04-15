load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xoc.ent", occ_1481_c3_p0_s0.8
hide everything, occ_1481_c3_p0_s0.8
show cartoon, occ_1481_c3_p0_s0.8 and chain D+A
color palegreen, occ_1481_c3_p0_s0.8 and chain D
color lightblue, occ_1481_c3_p0_s0.8 and chain A
select hotspot_source, occ_1481_c3_p0_s0.8 and ((chain D and resi 42))
select hotspot_target, occ_1481_c3_p0_s0.8 and ((chain A and resi 449))
select hotspot_all, occ_1481_c3_p0_s0.8 and ((chain A and resi 449) or (chain D and resi 42))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1481_c3_p0_s0.8 and chain D+A
set_name hotspot_all, hotspot_occurrence_1481
set_name hotspot_source, hotspot_source_1481
set_name hotspot_target, hotspot_target_1481
bg_color white
# patternId=0 support=0.8 graphId=360
