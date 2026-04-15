load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7ekh.ent", occ_24_c0_p0_s0.7
hide everything, occ_24_c0_p0_s0.7
show cartoon, occ_24_c0_p0_s0.7 and chain A+B
color palegreen, occ_24_c0_p0_s0.7 and chain A
color lightblue, occ_24_c0_p0_s0.7 and chain B
select hotspot_source, occ_24_c0_p0_s0.7 and ((chain A and resi 31))
select hotspot_target, occ_24_c0_p0_s0.7 and ((chain B and resi 455))
select hotspot_all, occ_24_c0_p0_s0.7 and ((chain A and resi 31) or (chain B and resi 455))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_24_c0_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_24
set_name hotspot_source, hotspot_source_24
set_name hotspot_target, hotspot_target_24
bg_color white
# patternId=0 support=0.7 graphId=135
