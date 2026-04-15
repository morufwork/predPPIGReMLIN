load "/mnt/f/research/cwork_hotspot/pdbfiles/pdb6lzg.ent", occ_201_c0_p0_s1.0
hide everything, occ_201_c0_p0_s1.0
show cartoon, occ_201_c0_p0_s1.0 and chain A+B
color palegreen, occ_201_c0_p0_s1.0 and chain A
color lightblue, occ_201_c0_p0_s1.0 and chain B
select hotspot_source, occ_201_c0_p0_s1.0 and ((chain A and resi 83))
select hotspot_target, occ_201_c0_p0_s1.0 and ((chain B and resi 486))
select hotspot_all, occ_201_c0_p0_s1.0 and ((chain A and resi 83) or (chain B and resi 486))
show sticks, hotspot_all
color tv_orange, hotspot_source
color hotpink, hotspot_target
show spheres, hotspot_all and name CA+C1*+C2*+P
set sphere_scale, 0.35, hotspot_all
zoom hotspot_all, 8
orient occ_201_c0_p0_s1.0 and chain A+B
set_name hotspot_all, hotspot_occurrence_201
set_name hotspot_source, hotspot_source_201
set_name hotspot_target, hotspot_target_201
bg_color white
# patternId=0 support=1.0 graphId=9
