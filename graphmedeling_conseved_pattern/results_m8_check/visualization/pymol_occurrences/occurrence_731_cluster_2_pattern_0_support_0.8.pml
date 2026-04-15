load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7xo6.ent", occ_731_c2_p0_s0.8
hide everything, occ_731_c2_p0_s0.8
show cartoon, occ_731_c2_p0_s0.8 and chain D+A
color palegreen, occ_731_c2_p0_s0.8 and chain D
color lightblue, occ_731_c2_p0_s0.8 and chain A
select hotspot_source, occ_731_c2_p0_s0.8 and ((chain D and resi 37))
select hotspot_target, occ_731_c2_p0_s0.8 and ((chain A and resi 505))
select hotspot_all, occ_731_c2_p0_s0.8 and ((chain A and resi 505) or (chain D and resi 37))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_731_c2_p0_s0.8 and chain D+A
set_name hotspot_all, hotspot_occurrence_731
set_name hotspot_source, hotspot_source_731
set_name hotspot_target, hotspot_target_731
bg_color white
# patternId=0 support=0.8 graphId=337
