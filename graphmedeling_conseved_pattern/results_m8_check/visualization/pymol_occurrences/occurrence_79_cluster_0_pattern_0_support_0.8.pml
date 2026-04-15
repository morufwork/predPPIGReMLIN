load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7efr.ent", occ_79_c0_p0_s0.8
hide everything, occ_79_c0_p0_s0.8
show cartoon, occ_79_c0_p0_s0.8 and chain A+B
color palegreen, occ_79_c0_p0_s0.8 and chain A
color lightblue, occ_79_c0_p0_s0.8 and chain B
select hotspot_source, occ_79_c0_p0_s0.8 and ((chain A and resi 27))
select hotspot_target, occ_79_c0_p0_s0.8 and ((chain B and resi 475))
select hotspot_all, occ_79_c0_p0_s0.8 and ((chain A and resi 27) or (chain B and resi 475))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_79_c0_p0_s0.8 and chain A+B
set_name hotspot_all, hotspot_occurrence_79
set_name hotspot_source, hotspot_source_79
set_name hotspot_target, hotspot_target_79
bg_color white
# patternId=0 support=0.8 graphId=70
