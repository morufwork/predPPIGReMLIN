load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7w9i.ent", occ_2083_c4_p0_s0.8
hide everything, occ_2083_c4_p0_s0.8
show cartoon, occ_2083_c4_p0_s0.8 and chain A+E
color palegreen, occ_2083_c4_p0_s0.8 and chain A
color lightblue, occ_2083_c4_p0_s0.8 and chain E
select hotspot_source, occ_2083_c4_p0_s0.8 and ((chain A and resi 27) or (chain A and resi 30))
select hotspot_target, occ_2083_c4_p0_s0.8 and ((chain E and resi 456))
select hotspot_all, occ_2083_c4_p0_s0.8 and ((chain A and resi 27) or (chain A and resi 30) or (chain E and resi 456))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_2083_c4_p0_s0.8 and chain A+E
set_name hotspot_all, hotspot_occurrence_2083
set_name hotspot_source, hotspot_source_2083
set_name hotspot_target, hotspot_target_2083
bg_color white
# patternId=0 support=0.8 graphId=250
