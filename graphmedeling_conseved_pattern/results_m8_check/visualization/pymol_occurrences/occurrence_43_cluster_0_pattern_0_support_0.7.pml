load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7wa1.ent", occ_43_c0_p0_s0.7
hide everything, occ_43_c0_p0_s0.7
show cartoon, occ_43_c0_p0_s0.7 and chain A+B
color palegreen, occ_43_c0_p0_s0.7 and chain A
color lightblue, occ_43_c0_p0_s0.7 and chain B
select hotspot_source, occ_43_c0_p0_s0.7 and ((chain A and resi 34))
select hotspot_target, occ_43_c0_p0_s0.7 and ((chain B and resi 453))
select hotspot_all, occ_43_c0_p0_s0.7 and ((chain A and resi 34) or (chain B and resi 453))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_43_c0_p0_s0.7 and chain A+B
set_name hotspot_all, hotspot_occurrence_43
set_name hotspot_source, hotspot_source_43
set_name hotspot_target, hotspot_target_43
bg_color white
# patternId=0 support=0.7 graphId=264
