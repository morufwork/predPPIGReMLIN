load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7efr.ent", occ_82_c0_p0_s0.8
hide everything, occ_82_c0_p0_s0.8
show cartoon, occ_82_c0_p0_s0.8 and chain A+B
color palegreen, occ_82_c0_p0_s0.8 and chain A
color lightblue, occ_82_c0_p0_s0.8 and chain B
select hotspot_source, occ_82_c0_p0_s0.8 and ((chain A and resi 330))
select hotspot_target, occ_82_c0_p0_s0.8 and ((chain B and resi 500))
select hotspot_all, occ_82_c0_p0_s0.8 and ((chain A and resi 330) or (chain B and resi 500))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_82_c0_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_82
set_name hotspot_source, hotspot_source_82
set_name hotspot_target, hotspot_target_82
bg_color white
# patternId=0 support=0.8 graphId=82
