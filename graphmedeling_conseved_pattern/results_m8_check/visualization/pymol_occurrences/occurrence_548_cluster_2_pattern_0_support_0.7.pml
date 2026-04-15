load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xoc.ent", occ_548_c2_p0_s0.7
hide everything, occ_548_c2_p0_s0.7
show cartoon, occ_548_c2_p0_s0.7 and chain D+A
color palegreen, occ_548_c2_p0_s0.7 and chain D
color lightblue, occ_548_c2_p0_s0.7 and chain A
select hotspot_source, occ_548_c2_p0_s0.7 and ((chain D and resi 37))
select hotspot_target, occ_548_c2_p0_s0.7 and ((chain A and resi 505))
select hotspot_all, occ_548_c2_p0_s0.7 and ((chain A and resi 505) or (chain D and resi 37))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_548_c2_p0_s0.7 and chain D+A
set_name hotspot_all, hotspot_occurrence_548
set_name hotspot_source, hotspot_source_548
set_name hotspot_target, hotspot_target_548
bg_color white
# patternId=0 support=0.7 graphId=356
