load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb7bh9.ent", occ_818_c2_p0_s0.9
hide everything, occ_818_c2_p0_s0.9
show cartoon, occ_818_c2_p0_s0.9 and chain A+E
color palegreen, occ_818_c2_p0_s0.9 and chain A
color lightblue, occ_818_c2_p0_s0.9 and chain E
select hotspot_source, occ_818_c2_p0_s0.9 and ((chain A and resi 38))
select hotspot_target, occ_818_c2_p0_s0.9 and ((chain E and resi 498))
select hotspot_all, occ_818_c2_p0_s0.9 and ((chain A and resi 38) or (chain E and resi 498))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_818_c2_p0_s0.9 and chain A+E
set_name hotspot_all, hotspot_occurrence_818
set_name hotspot_source, hotspot_source_818
set_name hotspot_target, hotspot_target_818
bg_color white
# patternId=0 support=0.9 graphId=28
