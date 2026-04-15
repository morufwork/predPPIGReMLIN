load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb8dma.ent", occ_1489_c3_p0_s0.8
hide everything, occ_1489_c3_p0_s0.8
show cartoon, occ_1489_c3_p0_s0.8 and chain A+D
color palegreen, occ_1489_c3_p0_s0.8 and chain A
color lightblue, occ_1489_c3_p0_s0.8 and chain D
select hotspot_source, occ_1489_c3_p0_s0.8 and ((chain A and resi 502))
select hotspot_target, occ_1489_c3_p0_s0.8 and ((chain D and resi 353))
select hotspot_all, occ_1489_c3_p0_s0.8 and ((chain A and resi 502) or (chain D and resi 353))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_1489_c3_p0_s0.8 and chain A+D
set_name hotspot_all, hotspot_occurrence_1489
set_name hotspot_source, hotspot_source_1489
set_name hotspot_target, hotspot_target_1489
bg_color white
# patternId=0 support=0.8 graphId=395
